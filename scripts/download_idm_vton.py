"""Download IDM-VTON (ECCV 2024) weights for offline use.

This pulls the auxiliary checkpoints (densepose, human parsing, openpose) and
the main SDXL try-on model from Hugging Face into:

  weights/idm-vton/          - diffusion model (unet, vae, text encoders, …)
  third_party/IDM-VTON/ckpt/ - preprocessing checkpoints

Expect ~25 GB total. Run while online:

    .venv\\Scripts\\python.exe scripts\\download_idm_vton.py

Resume is safe: partial files are kept and continued.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ["COVERSWAP_ALLOW_DOWNLOAD"] = "1"

from coverswap import config  # noqa: E402
from coverswap import net  # noqa: E402

REPO = "yisol/IDM-VTON"

# Hugging Face hosts both the diffusion stack and the preprocessing ckpts.
MODEL_FILES: list[tuple[str, Path, int, str]] = [
    ("model_index.json", config.IDM_VTON_WEIGHTS_DIR / "model_index.json", 500, "IDM index"),
    ("scheduler/scheduler_config.json", config.IDM_VTON_WEIGHTS_DIR / "scheduler" / "scheduler_config.json", 200, "scheduler"),
    ("tokenizer/tokenizer_config.json", config.IDM_VTON_WEIGHTS_DIR / "tokenizer" / "tokenizer_config.json", 200, "tokenizer"),
    ("tokenizer/vocab.json", config.IDM_VTON_WEIGHTS_DIR / "tokenizer" / "vocab.json", 1_000_000, "tokenizer vocab"),
    ("tokenizer/merges.txt", config.IDM_VTON_WEIGHTS_DIR / "tokenizer" / "merges.txt", 500_000, "tokenizer merges"),
    ("tokenizer/special_tokens_map.json", config.IDM_VTON_WEIGHTS_DIR / "tokenizer" / "special_tokens_map.json", 100, "tokenizer special"),
    ("tokenizer_2/tokenizer_config.json", config.IDM_VTON_WEIGHTS_DIR / "tokenizer_2" / "tokenizer_config.json", 200, "tokenizer 2"),
    ("tokenizer_2/vocab.json", config.IDM_VTON_WEIGHTS_DIR / "tokenizer_2" / "vocab.json", 1_000_000, "tokenizer 2 vocab"),
    ("tokenizer_2/merges.txt", config.IDM_VTON_WEIGHTS_DIR / "tokenizer_2" / "merges.txt", 500_000, "tokenizer 2 merges"),
    ("tokenizer_2/special_tokens_map.json", config.IDM_VTON_WEIGHTS_DIR / "tokenizer_2" / "special_tokens_map.json", 100, "tokenizer 2 special"),
    ("text_encoder/config.json", config.IDM_VTON_WEIGHTS_DIR / "text_encoder" / "config.json", 200, "text encoder"),
    (
        "text_encoder/model.safetensors",
        config.IDM_VTON_WEIGHTS_DIR / "text_encoder" / "model.safetensors",
        200_000_000,
        "text encoder weights",
    ),
    ("text_encoder_2/config.json", config.IDM_VTON_WEIGHTS_DIR / "text_encoder_2" / "config.json", 200, "text encoder 2"),
    (
        "text_encoder_2/model.safetensors",
        config.IDM_VTON_WEIGHTS_DIR / "text_encoder_2" / "model.safetensors",
        2_700_000_000,
        "text encoder 2 weights",
    ),
    ("image_encoder/config.json", config.IDM_VTON_WEIGHTS_DIR / "image_encoder" / "config.json", 200, "image encoder"),
    (
        "image_encoder/model.safetensors",
        config.IDM_VTON_WEIGHTS_DIR / "image_encoder" / "model.safetensors",
        2_400_000_000,
        "image encoder weights",
    ),
    ("vae/config.json", config.IDM_VTON_WEIGHTS_DIR / "vae" / "config.json", 200, "VAE"),
    (
        "vae/diffusion_pytorch_model.safetensors",
        config.IDM_VTON_WEIGHTS_DIR / "vae" / "diffusion_pytorch_model.safetensors",
        300_000_000,
        "VAE weights",
    ),
    ("unet/config.json", config.IDM_VTON_WEIGHTS_DIR / "unet" / "config.json", 200, "UNet"),
    (
        "unet/diffusion_pytorch_model.bin",
        config.IDM_VTON_WEIGHTS_DIR / "unet" / "diffusion_pytorch_model.bin",
        11_900_000_000,
        "UNet weights",
    ),
    ("unet_encoder/config.json", config.IDM_VTON_WEIGHTS_DIR / "unet_encoder" / "config.json", 200, "UNet encoder"),
    (
        "unet_encoder/diffusion_pytorch_model.safetensors",
        config.IDM_VTON_WEIGHTS_DIR / "unet_encoder" / "diffusion_pytorch_model.safetensors",
        10_200_000_000,
        "UNet encoder weights",
    ),
]

CKPT_FILES: list[tuple[str, Path, int, str]] = [
    (
        "densepose/model_final_162be9.pkl",
        config.IDM_VTON_CKPT_DIR / "densepose" / "model_final_162be9.pkl",
        200_000_000,
        "DensePose",
    ),
    (
        "humanparsing/parsing_atr.onnx",
        config.IDM_VTON_CKPT_DIR / "humanparsing" / "parsing_atr.onnx",
        200_000_000,
        "human parsing (ATR)",
    ),
    (
        "humanparsing/parsing_lip.onnx",
        config.IDM_VTON_CKPT_DIR / "humanparsing" / "parsing_lip.onnx",
        200_000_000,
        "human parsing (LIP)",
    ),
    (
        "openpose/ckpts/body_pose_model.pth",
        config.IDM_VTON_CKPT_DIR / "openpose" / "ckpts" / "body_pose_model.pth",
        150_000_000,
        "OpenPose body",
    ),
]

OPENPOSE_MIRROR = (
    "https://huggingface.co/lllyasviel/Annotators/resolve/main/body_pose_model.pth"
)


def _download_one(remote: str, dest: Path, min_bytes: int, label: str) -> bool:
    if dest.exists() and dest.stat().st_size >= min_bytes:
        print(f"  skip {label} ({dest.stat().st_size:,} bytes)", flush=True)
        return True
    url = net.hf_url(REPO, remote) if not remote.startswith("http") else remote
    ok = net.download(url, dest, progress=lambda m: print(f"  {m}", flush=True), label=label)
    if ok and dest.stat().st_size < min_bytes:
        print(f"  {label}: file too small ({dest.stat().st_size:,} < {min_bytes:,})", flush=True)
        return False
    return ok


def main() -> int:
    config.IDM_VTON_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    config.IDM_VTON_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print("IDM-VTON download (~25 GB). You can stop and resume anytime.", flush=True)
    failed: list[str] = []

    print("\n[1/2] Preprocessing checkpoints (densepose, parsing, openpose)...", flush=True)
    for remote, dest, min_bytes, label in CKPT_FILES:
        url = OPENPOSE_MIRROR if "body_pose_model" in dest.name else remote
        if not _download_one(url, dest, min_bytes, label):
            failed.append(label)

    print("\n[2/2] Diffusion model (SDXL try-on stack)...", flush=True)
    for remote, dest, min_bytes, label in MODEL_FILES:
        if not _download_one(remote, dest, min_bytes, label):
            failed.append(label)

    if failed:
        print(f"\nFAILED: {', '.join(failed)}", flush=True)
        return 1

    print("\nIDM-VTON download complete.", flush=True)
    print("Try: .venv\\Scripts\\python.exe scripts\\idm_probe.py", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
