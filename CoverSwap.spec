# PyInstaller spec for the CoverSwap desktop app.
#   .venv\Scripts\python.exe -m PyInstaller --noconfirm --clean CoverSwap.spec
#
# See the comment blocks below before changing excludes or binaries.
import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH)

datas, binaries, hiddenimports = collect_all("insightface")
datas += [
    (str(ROOT / "third_party" / "CatVTON" / "model"), "third_party/CatVTON/model"),
]

hiddenimports += collect_submodules("coverswap")
hiddenimports += [
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "ultralytics",
    "onnxruntime",
    "insightface",
    "einops",
]

# Do NOT --collect-all onnxruntime (analysis subprocess crashes on onnx.reference).
# matplotlib is required by insightface even if unused (~40 MB).
EXCLUDES = [
    "onnx.reference",
    "onnxruntime.transformers",
    "onnxruntime.quantization",
    "onnxruntime.tools",
    "onnxruntime.training",
    "onnxruntime.backend",
    "sympy",
    "mpmath",
    "pip",
    "tkinter",
    "PyQt6",
    "IPython",
    "pytest",
    "pyximport",
    "gradio",
]

# PyQt5 and scikit-learn ship duplicate MSVC runtimes under the same filename.
# Drop them so the process binds to the system runtime (required by onnxruntime).
STRIP_RUNTIME_DLLS = {
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll",
    "msvcp140_codecvt_ids.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "concrt140.dll",
}

a = Analysis(
    ["main.py"],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)

a.binaries = [
    b for b in a.binaries if os.path.basename(b[0]).lower() not in STRIP_RUNTIME_DLLS
]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="CoverSwap",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
