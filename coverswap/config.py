"""Central configuration: model choices, paths and device selection."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


def _project_root() -> Path:
    """Folder that holds weights/, outputs/, and the .exe when frozen."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def _bundle_root() -> Path:
    """Unpack location for bundled code (CatVTON sources, etc.) in a one-file build."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", _project_root()))
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = _project_root()
_BUNDLE_ROOT = _bundle_root()
WEIGHTS_DIR = PROJECT_ROOT / "weights"
THIRD_PARTY_DIR = _BUNDLE_ROOT / "third_party"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CATVTON_DIR = THIRD_PARTY_DIR / "CatVTON"
CATVTON_REPO_URL = "https://github.com/Zheng-Chong/CatVTON.git"
CATVTON_ZIP_URL = "https://codeload.github.com/Zheng-Chong/CatVTON/zip/refs/heads/main"

# IDM-VTON (ECCV 2024) — optional; ~25 GB under weights/idm-vton when installed.
IDM_VTON_DIR = THIRD_PARTY_DIR / "IDM-VTON"
IDM_VTON_GRADIO_DIR = IDM_VTON_DIR / "gradio_demo"
IDM_VTON_WEIGHTS_DIR = WEIGHTS_DIR / "idm-vton"
IDM_VTON_CKPT_DIR = IDM_VTON_DIR / "ckpt"
IDM_VTON_REPO_ID = "yisol/IDM-VTON"
IDM_VTON_ZIP_URL = "https://codeload.github.com/yisol/IDM-VTON/zip/refs/heads/main"

CLOTHES_BACKEND_CATVTON = "catvton"
CLOTHES_BACKEND_IDM = "idm-vton"
CLOTHES_BACKENDS = (CLOTHES_BACKEND_CATVTON, CLOTHES_BACKEND_IDM)

for _d in (WEIGHTS_DIR, THIRD_PARTY_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# huggingface_hub routes downloads through the hf_xet native client, which dies
# with an illegal instruction on some Windows builds as soon as a large file
# starts transferring. The pure-Python path is slower but does not crash.
# Set COVERSWAP_ALLOW_XET=1 to opt back in.
if os.environ.get("COVERSWAP_ALLOW_XET") != "1":
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# The app is meant to run with the internet unplugged. These flags stop
# Hugging Face / Diffusers / Ultralytics from reaching the network if a file
# is missing; we would rather fail with a clear message than hang.
# COVERSWAP_ALLOW_DOWNLOAD=1 is set only by the one-time installer.
if os.environ.get("COVERSWAP_ALLOW_DOWNLOAD") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("YOLO_OFFLINE", "1")

# Ultralytics downloads these from its own release assets on first use. Absolute
# paths keep them in weights/ instead of whatever the current directory happens
# to be. YOLO-pose gives boxes and keypoints in one pass; SAM 2 turns those
# boxes into pixel-accurate silhouettes. Neither needs a compiled CUDA
# extension, which is what makes this workable on Windows.
YOLO_POSE_MODEL = str(WEIGHTS_DIR / "yolo11m-pose.pt")
SAM_MODEL = str(WEIGHTS_DIR / "sam2.1_b.pt")

# CatVTON layers its attention weights onto an SD1.5-inpainting base.
#
# This repo is used instead of CatVTON's default because it publishes fp16
# safetensors: the UNet is 1.7GB rather than 3.4GB, and the pipeline runs in
# fp16 on this GPU anyway. Only the scheduler config and UNet are needed - the
# VAE comes from sd-vae-ft-mse (hardcoded upstream) and the safety checker is
# skipped, which avoids a further 1.2GB.
SD_INPAINT_REPO = "stable-diffusion-v1-5/stable-diffusion-inpainting"
BASE_MODEL_DIR = WEIGHTS_DIR / "sd15-inpaint-fp16"
BASE_MODEL_FILES = {
    "scheduler/scheduler_config.json": "scheduler/scheduler_config.json",
    "unet/config.json": "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors": "unet/diffusion_pytorch_model.safetensors",
}

CATVTON_REPO_ID = "zhengchong/CatVTON"
# Upstream maps the "mix" version name onto this directory.
CATVTON_ATTN_VERSION = "mix"
CATVTON_ATTN_DIR = "mix-48k-1024"
CATVTON_WEIGHTS_DIR = WEIGHTS_DIR / "catvton"
CATVTON_ATTN_FILES = {
    f"{CATVTON_ATTN_DIR}/attention/model.safetensors":
        f"{CATVTON_ATTN_DIR}/attention/model.safetensors",
}

# CatVTON hardcodes this VAE rather than using the base model's.
VAE_REPO = "stabilityai/sd-vae-ft-mse"
VAE_DIR = WEIGHTS_DIR / "sd-vae-ft-mse"
VAE_FILES = {
    "config.json": "config.json",
    "diffusion_pytorch_model.safetensors": "diffusion_pytorch_model.safetensors",
}

GARMENT_CATEGORIES = ("upper_body", "lower_body", "dress")

# Face swap (InsightFace + inswapper_128, from the swapping project)
FACESWAP_DIR = WEIGHTS_DIR / "faceswap"
FACESWAP_IF_ROOT = FACESWAP_DIR / "insightface"
FACESWAP_BUFFALO_DIR = FACESWAP_IF_ROOT / "models" / "buffalo_l"
FACESWAP_INSWAPPER = FACESWAP_DIR / "inswapper_128.onnx"
FACESWAP_BUFFALO_FILES = ("det_10g.onnx", "w600k_r50.onnx", "2d106det.onnx")
FACESWAP_BUFFALO_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
)
FACESWAP_INSWAPPER_MIRRORS = [
    "https://huggingface.co/hacksider/inswapper_128/resolve/main/inswapper_128.onnx",
    "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx",
    "https://huggingface.co/ModelsLab/inswapper/resolve/main/inswapper_128.onnx",
]

# App modes (sidebar tabs)
MODE_CLOTHES = "clothes"
MODE_FACE = "face"
MODE_BACKGROUND = "background"
SWAP_MODES = (MODE_CLOTHES, MODE_FACE, MODE_BACKGROUND)
MODE_LABELS = {
    MODE_CLOTHES: "Clothes",
    MODE_FACE: "Face",
    MODE_BACKGROUND: "Background",
}


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def resolve_device() -> str:
    """Pick a device, tolerating a torch-less environment.

    The window should still open if PyTorch is missing, so the user gets a
    readable message instead of a traceback before anything is drawn.
    """
    try:
        import torch
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def describe_device() -> str:
    try:
        import torch
    except Exception:
        return "PyTorch not installed - see requirements.txt"
    if not torch.cuda.is_available():
        return "CPU only (very slow; a CUDA GPU is strongly recommended)"
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return f"{name} - {total:.1f} GB"


@dataclass
class TryOnSettings:
    """Knobs exposed in the UI that affect a single try-on run."""

    backend: str = CLOTHES_BACKEND_CATVTON
    garment_description: str = "garment"
    category: str = "upper_body"
    steps: int = 30
    # Upstream suggests 2.5, but that noticeably washes out garment colour - a
    # black shirt came back olive grey. 4.0 reproduces the real colour and keeps
    # printed graphics legible.
    guidance_scale: float = 4.0
    seed: int = 42
    # CatVTON was trained at 1024x768. Defaulting to the smaller size roughly
    # halves runtime and peak VRAM, which matters on an 8GB laptop card that is
    # already sharing memory with the desktop compositor. The UI checkbox
    # switches up to the native resolution.
    width: int = 576
    height: int = 768
    mask_grow: int = 12
    feather: int = 9

    def as_dict(self) -> dict:
        return {
            "backend": self.backend,
            "category": self.category,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class FaceSwapSettings:
    tone_match: float = 0.7
    blend: float = 1.0


@dataclass
class BackgroundSwapSettings:
    feather: int = 15
    edge_grow: int = 8


@dataclass
class AppSettings:
    device: str = field(default_factory=lambda: os.environ.get("COVERSWAP_DEVICE", ""))
    low_vram: bool = True

    def device_or_auto(self) -> str:
        return self.device or resolve_device()
