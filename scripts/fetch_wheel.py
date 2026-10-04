"""Resumable, validated download of a single file (used for the big torch wheel).

Usage: python scripts/fetch_wheel.py <url> <dest>
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from coverswap import net  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2

    url, dest = sys.argv[1], Path(sys.argv[2])
    if not net.download(url, dest):
        print("FAILED", flush=True)
        return 1

    if dest.suffix in (".whl", ".zip"):
        try:
            with zipfile.ZipFile(dest) as archive:
                print(f"zip OK, {len(archive.namelist())} entries", flush=True)
        except Exception as exc:
            print(f"CORRUPT after download: {exc}", flush=True)
            return 1

    print("FETCH_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
