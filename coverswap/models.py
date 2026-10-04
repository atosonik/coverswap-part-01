"""What must be on disk for CoverSwap to run with the internet unplugged.

Runtime never downloads. The one-time `download_all()` helper is the only
place that talks to the network, and it is meant to be run once while online.
"""

from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from . import config, net

Progress = Optional[Callable[[str], None]]


@dataclass(frozen=True)
class ModelFile:
    """One file the app will refuse to start without."""

    key: str
    path: Path
    label: str
    min_bytes: int
    url: str = ""
    modes: tuple[str, ...] = (config.MODE_CLOTHES, config.MODE_FACE, config.MODE_BACKGROUND)


def _required() -> list[ModelFile]:
    detection = (
        config.MODE_CLOTHES,
        config.MODE_BACKGROUND,
    )
    return [
        ModelFile(
            key="yolo",
            path=Path(config.YOLO_POSE_MODEL),
            label="People finder",
            min_bytes=10_000_000,
            url="https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt",
            modes=detection,
        ),
        ModelFile(
            key="sam",
            path=Path(config.SAM_MODEL),
            label="Person outlines",
            min_bytes=50_000_000,
            url="https://github.com/ultralytics/assets/releases/download/v8.3.0/sam2.1_b.pt",
            modes=detection,
        ),
        ModelFile(
            key="scheduler",
            path=config.BASE_MODEL_DIR / "scheduler" / "scheduler_config.json",
            label="Try-on scheduler",
            min_bytes=200,
            url=net.hf_url(config.SD_INPAINT_REPO, "scheduler/scheduler_config.json"),
            modes=(config.MODE_CLOTHES,),
        ),
        ModelFile(
            key="unet_cfg",
            path=config.BASE_MODEL_DIR / "unet" / "config.json",
            label="Try-on model config",
            min_bytes=200,
            url=net.hf_url(config.SD_INPAINT_REPO, "unet/config.json"),
            modes=(config.MODE_CLOTHES,),
        ),
        ModelFile(
            key="unet",
            path=config.BASE_MODEL_DIR / "unet" / "diffusion_pytorch_model.safetensors",
            label="Clothes swap model",
            min_bytes=1_000_000_000,
            url=net.hf_url(
                config.SD_INPAINT_REPO, "unet/diffusion_pytorch_model.fp16.safetensors"
            ),
            modes=(config.MODE_CLOTHES,),
        ),
        ModelFile(
            key="vae_cfg",
            path=config.VAE_DIR / "config.json",
            label="Image decoder config",
            min_bytes=200,
            url=net.hf_url(config.VAE_REPO, "config.json"),
            modes=(config.MODE_CLOTHES,),
        ),
        ModelFile(
            key="vae",
            path=config.VAE_DIR / "diffusion_pytorch_model.safetensors",
            label="Image decoder",
            min_bytes=100_000_000,
            url=net.hf_url(config.VAE_REPO, "diffusion_pytorch_model.safetensors"),
            modes=(config.MODE_CLOTHES,),
        ),
        ModelFile(
            key="attn",
            path=config.CATVTON_WEIGHTS_DIR
            / config.CATVTON_ATTN_DIR
            / "attention"
            / "model.safetensors",
            label="Clothes details",
            min_bytes=50_000_000,
            url=net.hf_url(
                config.CATVTON_REPO_ID,
                f"{config.CATVTON_ATTN_DIR}/attention/model.safetensors",
            ),
            modes=(config.MODE_CLOTHES,),
        ),
        ModelFile(
            key="source",
            path=config.CATVTON_DIR / "model" / "pipeline.py",
            label="Try-on program files",
            min_bytes=1_000,
            url=config.CATVTON_ZIP_URL,
            modes=(config.MODE_CLOTHES,),
        ),
        ModelFile(
            key="inswapper",
            path=config.FACESWAP_INSWAPPER,
            label="Face swap model",
            min_bytes=500_000_000,
            modes=(config.MODE_FACE,),
        ),
        *[
            ModelFile(
                key=f"buffalo_{name}",
                path=config.FACESWAP_BUFFALO_DIR / name,
                label=f"Face detector ({name})",
                min_bytes=1_000_000,
                modes=(config.MODE_FACE,),
            )
            for name in config.FACESWAP_BUFFALO_FILES
        ],
    ]


def file_ok(item: ModelFile) -> bool:
    return item.path.is_file() and item.path.stat().st_size >= item.min_bytes


def missing_models(mode: str | None = None) -> list[ModelFile]:
    items = _required()
    if mode:
        items = [item for item in items if mode in item.modes]
    return [item for item in items if not file_ok(item)]


def ready(mode: str | None = None) -> bool:
    return not missing_models(mode)


def ready_for_mode(mode: str) -> bool:
    return ready(mode)


def summary() -> str:
    items = _required()
    present = [item for item in items if file_ok(item)]
    total = sum(item.path.stat().st_size for item in present)
    clothes = ready(config.MODE_CLOTHES)
    face = ready(config.MODE_FACE)
    bg = ready(config.MODE_BACKGROUND)
    return (
        f"{len(present)}/{len(items)} files ({total / 1e9:.1f} GB) · "
        f"clothes {'OK' if clothes else 'missing'} · "
        f"face {'OK' if face else 'missing'} · "
        f"background {'OK' if bg else 'missing'}"
    )


def offline_error(item: ModelFile | None = None) -> RuntimeError:
    extra = f" Missing: {item.label}." if item else ""
    return RuntimeError(
        "CoverSwap is set up to work without the internet, but a model file "
        "is not on this computer." + extra + " Connect to the internet once and click "
        "“Download models”, or run scripts\\download_models.py."
    )


def require_ready(mode: str | None = None) -> None:
    missing = missing_models(mode)
    if missing:
        raise offline_error(missing[0])


def _download_buffalo(progress: Progress = None) -> None:
    if all((config.FACESWAP_BUFFALO_DIR / f).is_file() for f in config.FACESWAP_BUFFALO_FILES):
        return

    cached = Path.home() / ".insightface" / "models" / "buffalo_l"
    if cached.is_dir() and all((cached / f).is_file() for f in config.FACESWAP_BUFFALO_FILES):
        config.FACESWAP_BUFFALO_DIR.mkdir(parents=True, exist_ok=True)
        for name in config.FACESWAP_BUFFALO_FILES:
            shutil.copy2(cached / name, config.FACESWAP_BUFFALO_DIR / name)
        return

    zip_path = config.FACESWAP_DIR / "buffalo_l.zip"
    config.FACESWAP_BUFFALO_DIR.mkdir(parents=True, exist_ok=True)
    if not net.download(config.FACESWAP_BUFFALO_URL, zip_path, progress=progress, label="Face pack"):
        raise RuntimeError("Could not download the face detection pack.")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(config.FACESWAP_BUFFALO_DIR)
    zip_path.unlink(missing_ok=True)


def _download_inswapper(progress: Progress = None) -> None:
    if config.FACESWAP_INSWAPPER.is_file() and config.FACESWAP_INSWAPPER.stat().st_size > 500_000_000:
        try:
            import onnx

            onnx.load(str(config.FACESWAP_INSWAPPER))
            return
        except Exception:
            config.FACESWAP_INSWAPPER.unlink(missing_ok=True)

    config.FACESWAP_DIR.mkdir(parents=True, exist_ok=True)
    for url in config.FACESWAP_INSWAPPER_MIRRORS:
        if config.FACESWAP_INSWAPPER.exists():
            config.FACESWAP_INSWAPPER.unlink()
        if net.download(url, config.FACESWAP_INSWAPPER, progress=progress, label="Face swap model"):
            if config.FACESWAP_INSWAPPER.stat().st_size > 500_000_000:
                try:
                    import onnx

                    onnx.load(str(config.FACESWAP_INSWAPPER))
                    return
                except Exception:
                    config.FACESWAP_INSWAPPER.unlink(missing_ok=True)
    raise RuntimeError("Could not download inswapper_128.onnx from any mirror.")


def download_all(progress: Progress = None) -> None:
    """Fetch every missing file. Safe to re-run; complete files are skipped."""

    def note(message: str) -> None:
        if progress:
            progress(message)

    items = _required()
    todo = [item for item in items if not file_ok(item)]
    if not todo:
        note("All models are already on this computer.")
        return

    note(f"Downloading {len(todo)} file(s). One-time setup, about 3 GB total.")
    for item in todo:
        if item.key.startswith("buffalo_"):
            continue
        if item.key == "inswapper":
            continue

        item.path.parent.mkdir(parents=True, exist_ok=True)
        if item.key == "source":
            if not net.download_and_extract_zip(item.url, config.CATVTON_DIR, progress=progress):
                raise RuntimeError(
                    "Could not download the try-on program files. Check the connection and try again."
                )
            continue
        if not item.url:
            raise RuntimeError(f"No download address for {item.label}.")
        if not net.download(item.url, item.path, progress=progress, label=item.label):
            item.path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Could not download {item.label}. Check the connection and try again."
            )
        if not file_ok(item):
            raise RuntimeError(f"{item.label} downloaded but looks incomplete.")

    if any(item.key.startswith("buffalo_") for item in todo):
        note("Downloading face detection pack…")
        _download_buffalo(progress)
    if any(item.key == "inswapper" for item in todo):
        note("Downloading face swap model…")
        _download_inswapper(progress)

    if missing_models():
        raise RuntimeError("Download finished, but some files are still missing.")
    note("All models are ready. You can unplug the internet.")
