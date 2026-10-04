"""Stage 3: the actual garment swap, backed by CatVTON.

CatVTON is the practical choice on an 8GB card: the whole model is ~899M
parameters and inference fits under 8GB at 1024x768, where FLUX-based try-on
models need several times that.

Only `CatVTONPipeline` is imported from the upstream repo. Its `cloth_masker`
module - the part that pulls in DensePose and Detectron2 - is deliberately left
alone, because `clothing.py` produces the mask instead.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image

from . import config, models

Progress = Optional[Callable[[str], None]]


def _report(cb: Progress, message: str) -> None:
    if cb:
        cb(message)


@contextlib.contextmanager
def _local_from_pretrained():
    """Force every Diffusers `from_pretrained` to stay on disk.

    Without this, a missing sidecar file would quietly reach Hugging Face even
    though the user asked for a fully offline app.
    """
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel

    patches = []
    for cls in (AutoencoderKL, DDIMScheduler, UNet2DConditionModel):
        original = cls.from_pretrained

        def wrapped(*args, _original=original, **kwargs):
            kwargs["local_files_only"] = True
            return _original(*args, **kwargs)

        cls.from_pretrained = wrapped
        patches.append((cls, original))
    try:
        yield
    finally:
        for cls, original in patches:
            cls.from_pretrained = original


@contextlib.contextmanager
def _vae_from(local_dir: Path):
    """Redirect `AutoencoderKL.from_pretrained` to a local directory.

    CatVTON hardcodes the VAE repo id, so the only way to load it from disk is
    to intercept the call for the duration of construction.
    """
    from diffusers import AutoencoderKL

    original = AutoencoderKL.from_pretrained

    def patched(path, *args, **kwargs):
        if str(path) == config.VAE_REPO:
            path = str(local_dir)
        kwargs["local_files_only"] = True
        return original(path, *args, **kwargs)

    AutoencoderKL.from_pretrained = patched
    try:
        yield
    finally:
        AutoencoderKL.from_pretrained = original


def pad_to_ratio(image: Image.Image, width: int, height: int, fill=(255, 255, 255)) -> Image.Image:
    """Letterbox a garment photo onto the model's aspect ratio.

    Stretching a flat-lay to fit distorts prints and logos, which is exactly
    the detail the model is being asked to reproduce.
    """
    target_ratio = width / height
    w, h = image.size
    ratio = w / h
    if ratio > target_ratio:
        new_h = int(w / target_ratio)
        canvas = Image.new("RGB", (w, new_h), fill)
        canvas.paste(image, (0, (new_h - h) // 2))
    else:
        new_w = int(h * target_ratio)
        canvas = Image.new("RGB", (new_w, h), fill)
        canvas.paste(image, ((new_w - w) // 2, 0))
    return canvas.resize((width, height), Image.LANCZOS)


def fit_person(image: Image.Image, width: int, height: int) -> Image.Image:
    """Scale a person crop to the model resolution, padding rather than cropping
    so no part of the body is lost."""
    return pad_to_ratio(image, width, height, fill=(127, 127, 127))


class CatVTONBackend:
    """Wraps the upstream pipeline and keeps it resident between runs."""

    def __init__(self, device: Optional[str] = None, low_vram: bool = True) -> None:
        self.device = device or config.resolve_device()
        self.low_vram = low_vram
        self._pipeline = None

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    def load(self, progress: Progress = None):
        if self._pipeline is not None:
            return self._pipeline

        import torch

        models.require_ready(config.MODE_CLOTHES)
        repo_dir = config.CATVTON_DIR
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))

        _report(progress, "Loading the clothes model... first time takes about 20 seconds.")
        from model.pipeline import CatVTONPipeline  # type: ignore[import-not-found]

        with _local_from_pretrained(), _vae_from(config.VAE_DIR):
            self._pipeline = CatVTONPipeline(
                base_ckpt=str(config.BASE_MODEL_DIR),
                attn_ckpt=str(config.CATVTON_WEIGHTS_DIR),
                attn_ckpt_version=config.CATVTON_ATTN_VERSION,
                weight_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device=self.device,
                skip_safety_check=True,
                use_tf32=True,
            )
        if self.low_vram and self.device == "cuda":
            # Slicing trades a little speed for headroom, which is the right
            # call when the desktop compositor is already holding ~3GB.
            try:
                self._pipeline.vae.enable_slicing()
            except AttributeError:
                pass
        _report(progress, "Model ready.")
        return self._pipeline

    def unload(self) -> None:
        self._pipeline = None
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
        mask: Image.Image,
        settings: config.TryOnSettings,
        progress: Progress = None,
    ) -> Image.Image:
        """Swap one garment onto one person crop."""
        import torch

        pipeline = self.load(progress)
        w, h = settings.width, settings.height

        person_in = fit_person(person.convert("RGB"), w, h)
        garment_in = pad_to_ratio(garment.convert("RGB"), w, h)
        mask_in = pad_to_ratio(mask.convert("RGB"), w, h, fill=(0, 0, 0)).convert("L")

        generator = torch.Generator(device=self.device).manual_seed(int(settings.seed))
        _report(progress, "Drawing the new clothes… almost done.")
        result = pipeline(
            image=person_in,
            condition_image=garment_in,
            mask=mask_in,
            # Without these the pipeline silently falls back to its 1024x768
            # defaults and the result no longer matches the prepared inputs.
            height=h,
            width=w,
            num_inference_steps=int(settings.steps),
            guidance_scale=float(settings.guidance_scale),
            generator=generator,
        )[0]
        return unpad(result, person.size, (w, h))


def unpad(result: Image.Image, original_size: tuple[int, int], model_size: tuple[int, int]) -> Image.Image:
    """Undo `fit_person` padding and return to the crop's original pixel size."""
    ow, oh = original_size
    mw, mh = model_size
    target_ratio = mw / mh
    ratio = ow / oh

    if ratio > target_ratio:
        content_h = int(mh * (target_ratio / ratio))
        top = (mh - content_h) // 2
        result = result.crop((0, top, mw, top + content_h))
    else:
        content_w = int(mw * (ratio / target_ratio))
        left = (mw - content_w) // 2
        result = result.crop((left, 0, left + content_w, mh))

    return result.resize((ow, oh), Image.LANCZOS)


def to_pil(array: np.ndarray) -> Image.Image:
    """BGR ndarray (OpenCV) -> RGB PIL."""
    return Image.fromarray(array[..., ::-1])


def to_bgr(image: Image.Image) -> np.ndarray:
    """RGB PIL -> BGR ndarray (OpenCV)."""
    return np.array(image.convert("RGB"))[..., ::-1].copy()
