# CoverSwap

Offline desktop app for editing people in photos: **change clothes**, **swap faces**, or **replace the background**. Everything runs locally — no cloud, no accounts, no telemetry. After models are on disk you can unplug the network.

Built with PyQt5. Clothes swapping uses [CatVTON](https://github.com/Zheng-Chong/CatVTON) on Stable Diffusion 1.5 inpainting (~3 GB of weights, fits 8 GB VRAM). Face swap uses InsightFace + `inswapper_128.onnx` (same stack as the Face Swap sibling project). Background replacement uses YOLO pose + SAM 2 person masks.

## What you can do

| Tab | Input | What happens |
|-----|--------|--------------|
| **Clothes** | Photo + garment image | Detect people → mask clothing region → CatVTON try-on → composite |
| **Face** | Photo + replacement face | Detect faces → inswapper swap with tone/blend controls |
| **Background** | Photo + new background | Detect people → silhouette mask → replace everything behind them |

Click a person or face on the canvas (or pick from the list) before running a swap.

## First run after cloning

Model files are **not** stored as single large blobs in git. They are committed as ≤1 MB parts under `parts/`, checksummed in `manifest.json`. Reassemble them first:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python chunker.py merge          # parts/ -> weights/  (~3.2 GB, ~20 s, checksummed)
python main.py                   # or: run.bat
```

If `parts/` is missing, download models while online instead:

```powershell
python scripts\download_models.py
```

Optional sample photos (family + clothes) for testing:

```powershell
python scripts\download_samples.py
```

## Models and disk space

| Purpose | Files under `weights/` | Approx. size |
|---------|------------------------|--------------|
| People detection | `yolo11m-pose.pt` | 42 MB |
| Person outlines (SAM 2) | `sam2.1_b.pt` | 162 MB |
| Clothes swap (SD1.5 UNet fp16) | `sd15-inpaint-fp16/unet/…` | 1.7 GB |
| Image decoder (VAE) | `sd-vae-ft-mse/…` | 335 MB |
| Garment details (CatVTON attn) | `catvton/mix-48k-1024/attention/…` | 198 MB |
| Face swap | `faceswap/inswapper_128.onnx` | 554 MB |
| Face detection / embedding | `faceswap/insightface/models/buffalo_l/*.onnx` | ~196 MB |

**Total: ~3.2 GB** in `weights/` after merge.

CatVTON **source code** lives in `third_party/CatVTON/` (bundled inside the `.exe`; fetched by `download_models.py` if missing).

### Optional: IDM-VTON (not included in `parts/`)

[IDM-VTON](https://github.com/yisol/IDM-VTON) (ECCV 2024) is wired in for experiments but **not** part of the default bundle (~25 GB extra, 16 GB+ VRAM recommended). To try it later:

```powershell
python scripts\download_idm_vton.py   # only when you explicitly want this
python scripts\idm_probe.py
```

## What is in the repo

| Path | Committed? | Notes |
|------|------------|-------|
| `coverswap/`, `main.py`, `chunker.py`, `README.md` | yes | application code |
| `CoverSwap.spec` | yes | hand-written PyInstaller spec — do not auto-generate over it |
| `parts/` | yes | model weights as ≤1 MB chunks — `chunker.py merge` restores `weights/` (~3.2 GB; first `git add` of this folder takes a few minutes) |
| `runableparts/` | no | local split of `CoverSwap.exe` (~3 GB). Rebuild; do not commit |
| `third_party/CatVTON/` | yes* | try-on pipeline sources (*or downloaded on first setup) |
| `third_party/IDM-VTON/` | no | optional clone + checkpoints; gitignored |
| `scripts/` | yes | download helpers, probes, release build |
| `weights/`, `dist/`, `build/` | no | regenerated locally |
| `outputs/`, `samples/`, `.venv/` | no | scratch / optional test images |

## Install (Python)

Verified on **Windows 11 + Python 3.12** with an **NVIDIA GPU** (CUDA). CPU-only works but clothes generation is very slow.

1. Create a venv and install PyTorch with CUDA (adjust the index URL for your driver/CUDA version).
2. `pip install -r requirements.txt`
3. Merge parts or run `scripts\download_models.py`.

`insightface==0.7.3` may compile from source; if pip fails, install Visual Studio “Desktop development with C++” and retry.

**Windows note:** `main.py` imports `onnxruntime` before PyQt5 on purpose — reversing that order can break DLL loading on some machines.

## Run

```powershell
python main.py
# or
run.bat
```

### Probe scripts (headless checks)

```powershell
python scripts\detect_probe.py [photo]
python scripts\mask_probe.py [photo] [category]
python scripts\tryon_probe.py [photo] [garment] [category]
python scripts\app_smoke.py
```

Results land in `outputs/`.

## Offline behaviour

At import time the app sets `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, and `YOLO_OFFLINE` unless `COVERSWAP_ALLOW_DOWNLOAD=1` (used only by the download scripts). Runtime code never fetches weights from the network — missing files produce a clear error pointing you to `scripts\download_models.py`.

## Hardware

| Task | Practical minimum |
|------|-------------------|
| Face swap | CPU ok; GPU helps detection only |
| Background swap | CPU ok |
| Clothes (CatVTON) | **8 GB VRAM** recommended; default resolution 576×768 to save memory |

The UI exposes a higher-resolution option (1024×768 native CatVTON size) if you have headroom.

## The standalone `.exe`

The one-file Windows build is about **3 GB** because it embeds PyTorch. That is too large to keep in git (thousands of 1 MB chunks, slow `git add`, oversized GitHub repo). Build it locally:

```powershell
scripts\build_release.bat
```

That writes `dist\CoverSwap.exe` and can also split it to local `runableparts/` for USB/copy, but those chunks stay untracked.

Working folder layout:

```
CoverSwap.exe
weights/          <- from: python chunker.py merge
outputs/          <- created when you save results
```

The exe needs **`weights/` next to it**. CatVTON code is embedded in the exe; only weights stay external.

The exe runs with **console visible** on purpose — a one-file GUI that fails during import has no window to show errors. First launch extracts ~3 GB to a temp folder (normal one-file cost).

Read the comment blocks in [CoverSwap.spec](CoverSwap.spec) before editing it:

- Do **not** `--collect-all onnxruntime` (analysis crashes on `onnx.reference`).
- Do **not** exclude `matplotlib` (InsightFace imports it at load time).
- Bundled MSVC runtime DLLs from PyQt5/sklearn are **stripped** so the system runtime is used (required for onnxruntime inside PyInstaller).

## Splitting large files (`chunker.py`)

Adapted from the Face Swap project. Splits any file or folder into ≤1 MB parts with per-part and whole-file SHA-256 in `manifest.json`. `merge` verifies every byte before writing; corrupted parts are rejected.

```powershell
python chunker.py split                              # weights/ -> parts/
python chunker.py split --src dist\CoverSwap.exe --parts runableparts
python chunker.py merge                                # parts/ -> weights/
python chunker.py merge --parts runableparts           # -> dist\CoverSwap.exe
python chunker.py verify                               # checksum parts/, no writes
python chunker.py verify --parts runableparts
```

Use `--all` to split every file under `weights/`, not just the 12 files the app loads. Use `--force` on merge to rebuild files that already exist.

## Project layout

```
coverswap/           Python package (UI, pipeline, models inventory)
main.py              Entry point
chunker.py           Split / merge / verify weights and exe
CoverSwap.spec       PyInstaller one-file spec
parts/               Model weight chunks (committed)
runableparts/        Local CoverSwap.exe chunks (gitignored)
weights/             Assembled models (gitignored, ~3.2 GB)
third_party/         CatVTON sources (IDM-VTON clone is gitignored)
scripts/             Downloads, probes, build_release.bat
outputs/             Saved results (gitignored)
```

## Pipeline (clothes tab)

```
photo  →  YOLO11 pose (boxes + keypoints)
       →  SAM 2 (pixel-accurate person mask)
       →  clothing mask for category (upper / lower / dress)
       →  CatVTON diffusion try-on
       →  composite back onto original photo
```

Face and background tabs reuse detection where needed and skip CatVTON entirely.

## Licence and upstream

- CatVTON, IDM-VTON, InsightFace, Ultralytics YOLO/SAM — see their respective repos and licences.
- CoverSwap application code: use and modify for personal/offline use; respect the non-commercial terms of bundled model weights where applicable.
