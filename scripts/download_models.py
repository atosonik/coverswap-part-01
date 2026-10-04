"""One-time download of every AI model CoverSwap needs.

Run this while you have internet. After it finishes you can unplug and use
the app normally:

    .venv\\Scripts\\python.exe scripts\\download_models.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# The app sets these so runtime cannot reach the network. They must be cleared
# here or the download itself would be blocked.
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ["COVERSWAP_ALLOW_DOWNLOAD"] = "1"


def main() -> int:
    from coverswap import models

    print(models.summary(), flush=True)
    if models.ready():
        print("Nothing to download. You can run CoverSwap offline.")
        return 0
    try:
        models.download_all(progress=lambda m: print(m, flush=True))
    except Exception as exc:
        print(f"FAILED: {exc}", flush=True)
        return 1
    print(models.summary(), flush=True)
    print("DONE. You can now use CoverSwap without internet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
