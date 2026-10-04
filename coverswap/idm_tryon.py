"""IDM-VTON (ECCV 2024) clothes swap backend.

Wraps the upstream Gradio inference path with local weight paths and
sequential CPU offloading so it can run on an 8 GB laptop GPU (slowly).
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from PIL import Image

from . import config

Progress = Optional[Callable[[str], None]]

IDM_ROOT = config.IDM_VTON_DIR
GRADIO_DIR = config.IDM_VTON_GRADIO_DIR
WEIGHTS = config.IDM_VTON_WEIGHTS_DIR


def _report(cb: Progress, message: str) -> None:
    if cb:
        cb(message)


def _ensure_import_paths() -> None:
    for path in (IDM_ROOT, GRADIO_DIR):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


@contextlib.contextmanager
def _local_from_pretrained():
    """Force Diffusers / Transformers to load from disk only."""
    from diffusers import AutoencoderKL, DDPMScheduler
    from transformers import (
        AutoTokenizer,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPVisionModelWithProjection,
    )

    patched = []
    for cls in (
        AutoencoderKL,
        DDPMScheduler,
        AutoTokenizer,
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPVisionModelWithProjection,
    ):
        original = cls.from_pretrained

        def wrapped(*args, _original=original, **kwargs):
            if args and isinstance(args[0], str) and args[0] in ("yisol/IDM-VTON", config.IDM_VTON_REPO_ID):
                args = (str(WEIGHTS), *args[1:])
            kwargs["local_files_only"] = True
            return _original(*args, **kwargs)

        cls.from_pretrained = wrapped
        patched.append((cls, original))

    from src import unet_hacked_garmnet, unet_hacked_tryon  # type: ignore[import-not-found]

    for module, name in (
        (unet_hacked_tryon, "UNet2DConditionModel"),
        (unet_hacked_garmnet, "UNet2DConditionModel"),
    ):
        cls = getattr(module, name)
        original = cls.from_pretrained

        def wrapped(*args, _original=original, **kwargs):
            if args and isinstance(args[0], str) and args[0] in ("yisol/IDM-VTON", config.IDM_VTON_REPO_ID):
                args = (str(WEIGHTS), *args[1:])
            kwargs["local_files_only"] = True
            return _original(*args, **kwargs)

        cls.from_pretrained = wrapped
        patched.append((cls, original))

    try:
        yield
    finally:
        for cls, original in patched:
            cls.from_pretrained = original


def _ckpt_ready() -> bool:
    checks = (
        config.IDM_VTON_CKPT_DIR / "densepose" / "model_final_162be9.pkl",
        config.IDM_VTON_CKPT_DIR / "humanparsing" / "parsing_atr.onnx",
        config.IDM_VTON_CKPT_DIR / "humanparsing" / "parsing_lip.onnx",
        config.IDM_VTON_CKPT_DIR / "openpose" / "ckpts" / "body_pose_model.pth",
        WEIGHTS / "unet" / "diffusion_pytorch_model.bin",
        WEIGHTS / "unet_encoder" / "diffusion_pytorch_model.safetensors",
    )
    return all(p.exists() and p.stat().st_size > 1_000_000 for p in checks)


class IDMVTONBackend:
    """Lazy-loaded IDM-VTON pipeline."""

    def __init__(self, device: Optional[str] = None, low_vram: bool = True) -> None:
        self.device = device or config.resolve_device()
        self.low_vram = low_vram
        self._pipe = None
        self._parsing = None
        self._openpose = None
        self._tensor_transform = None
        self._cwd: Optional[str] = None

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    def load(self, progress: Progress = None):
        if self._pipe is not None:
            return self._pipe

        if not _ckpt_ready():
            raise RuntimeError(
                "IDM-VTON weights are missing. Run: .venv\\Scripts\\python.exe scripts\\download_idm_vton.py"
            )

        import torch
        from diffusers import AutoencoderKL, DDPMScheduler
        from torchvision import transforms
        from transformers import (
            AutoTokenizer,
            CLIPImageProcessor,
            CLIPTextModel,
            CLIPTextModelWithProjection,
            CLIPVisionModelWithProjection,
        )

        _ensure_import_paths()
        _report(progress, "Loading IDM-VTON (first run takes a few minutes)...")

        from preprocess.humanparsing.run_parsing import Parsing  # type: ignore[import-not-found]
        from preprocess.openpose.run_openpose import OpenPose  # type: ignore[import-not-found]
        from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline  # type: ignore[import-not-found]
        from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref  # type: ignore[import-not-found]
        from src.unet_hacked_tryon import UNet2DConditionModel  # type: ignore[import-not-found]

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        base = str(WEIGHTS)

        with _local_from_pretrained():
            unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", torch_dtype=dtype)
            unet_encoder = UNet2DConditionModel_ref.from_pretrained(
                base, subfolder="unet_encoder", torch_dtype=dtype
            )
            tokenizer_one = AutoTokenizer.from_pretrained(base, subfolder="tokenizer", use_fast=False)
            tokenizer_two = AutoTokenizer.from_pretrained(base, subfolder="tokenizer_2", use_fast=False)
            noise_scheduler = DDPMScheduler.from_pretrained(base, subfolder="scheduler")
            text_encoder_one = CLIPTextModel.from_pretrained(
                base, subfolder="text_encoder", torch_dtype=dtype
            )
            text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
                base, subfolder="text_encoder_2", torch_dtype=dtype
            )
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                base, subfolder="image_encoder", torch_dtype=dtype
            )
            vae = AutoencoderKL.from_pretrained(base, subfolder="vae", torch_dtype=dtype)
            pipe = TryonPipeline.from_pretrained(
                base,
                unet=unet,
                vae=vae,
                feature_extractor=CLIPImageProcessor(),
                text_encoder=text_encoder_one,
                text_encoder_2=text_encoder_two,
                tokenizer=tokenizer_one,
                tokenizer_2=tokenizer_two,
                scheduler=noise_scheduler,
                image_encoder=image_encoder,
                torch_dtype=dtype,
            )
        pipe.unet_encoder = unet_encoder

        for module in (
            unet,
            unet_encoder,
            vae,
            text_encoder_one,
            text_encoder_two,
            image_encoder,
            pipe.unet,
            pipe.unet_encoder,
            pipe.vae,
            pipe.text_encoder,
            pipe.text_encoder_2,
            pipe.image_encoder,
        ):
            module.requires_grad_(False)

        if self.low_vram and self.device == "cuda":
            pipe.enable_sequential_cpu_offload(gpu_id=0)
        else:
            pipe.to(self.device)

        gpu_id = 0 if self.device == "cuda" else 0
        self._pipe = pipe
        self._parsing = Parsing(gpu_id)
        self._openpose = OpenPose(gpu_id)
        self._tensor_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )
        self._cwd = os.getcwd()
        _report(progress, "IDM-VTON ready.")
        return self._pipe

    def unload(self) -> None:
        self._pipe = None
        self._parsing = None
        self._openpose = None
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def run(
        self,
        person: Image.Image,
        garment: Image.Image,
        category: str,
        garment_description: str,
        steps: int = 30,
        guidance_scale: float = 2.0,
        seed: int = 42,
        width: int = 768,
        height: int = 1024,
        progress: Progress = None,
    ) -> Image.Image:
        import torch
        from detectron2.data.detection_utils import (  # type: ignore[import-not-found]
            _apply_exif_orientation,
            convert_PIL_to_numpy,
        )
        from torchvision.transforms.functional import to_pil_image

        import apply_net  # type: ignore[import-not-found]
        from utils_mask import get_mask_location  # type: ignore[import-not-found]

        _ensure_import_paths()
        pipe = self.load(progress)

        garm_img = garment.convert("RGB").resize((width, height))
        human_img = person.convert("RGB").resize((width, height))

        _report(progress, "Building body mask (OpenPose + parsing)...")
        self._openpose.preprocessor.body_estimation.model.to(self.device if self.device == "cuda" else "cpu")
        keypoints = self._openpose(human_img.resize((384, 512)))
        model_parse, _ = self._parsing(human_img.resize((384, 512)))
        mask, _mask_gray = get_mask_location("hd", category, model_parse, keypoints)
        mask = mask.resize((width, height))

        _report(progress, "Running DensePose...")
        prev_cwd = os.getcwd()
        os.chdir(IDM_ROOT)
        try:
            human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
            human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")
            args = apply_net.create_argument_parser().parse_args(
                (
                    "show",
                    "./configs/densepose_rcnn_R_50_FPN_s1x.yaml",
                    "./ckpt/densepose/model_final_162be9.pkl",
                    "dp_segm",
                    "-v",
                    "--opts",
                    "MODEL.DEVICE",
                    "cuda" if self.device == "cuda" else "cpu",
                )
            )
            pose_img = args.func(args, human_img_arg)
        finally:
            os.chdir(prev_cwd)

        pose_img = pose_img[:, :, ::-1]
        pose_img = Image.fromarray(pose_img).resize((width, height))

        garment_des = garment_description.strip() or "garment"
        _report(progress, "Generating try-on (this may take several minutes on 8 GB VRAM)...")

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device == "cuda"):
            prompt = "model is wearing " + garment_des
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(
                prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )

            cloth_prompt = "a photo of " + garment_des
            cloth_negative = negative_prompt
            cloth_prompts: List[str] = [cloth_prompt]
            cloth_negs: List[str] = [cloth_negative]
            (prompt_embeds_c, _, _, _) = pipe.encode_prompt(
                cloth_prompts,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=cloth_negs,
            )

            pose_tensor = self._tensor_transform(pose_img).unsqueeze(0)
            garm_tensor = self._tensor_transform(garm_img).unsqueeze(0)
            generator = torch.Generator(self.device).manual_seed(int(seed))

            images = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=int(steps),
                generator=generator,
                strength=1.0,
                pose_img=pose_tensor,
                text_embeds_cloth=prompt_embeds_c,
                cloth=garm_tensor,
                mask_image=mask,
                image=human_img,
                height=height,
                width=width,
                ip_adapter_image=garm_img,
                guidance_scale=float(guidance_scale),
            )[0]

        return images[0].resize(person.size, Image.LANCZOS)
