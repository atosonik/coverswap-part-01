"""Resumable, validated HTTP downloads.

Large downloads on an unreliable connection need more care than a plain GET.
`curl -C -` in particular is unsafe: when a server answers a Range request with
a full 200 body, curl appends it to the partial file and silently corrupts it -
which is how a 2.6GB wheel ended up bigger than its own Content-Length.

Every response here is checked (status and Content-Range) before any bytes are
written, and the finished file is verified against Content-Length.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Optional

CHUNK = 1 << 20
TIMEOUT = 30
UA = {"User-Agent": "coverswap/1.0"}

Progress = Optional[Callable[[str], None]]


def _report(cb: Progress, message: str) -> None:
    if cb:
        cb(message)
    else:
        print(message, flush=True)


def hf_url(repo_id: str, filename: str, revision: str = "main") -> str:
    """Direct download URL for a file in a Hugging Face repo."""
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"


def content_length(url: str) -> int:
    request = urllib.request.Request(url, headers=UA, method="HEAD")
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return int(response.headers["Content-Length"])


def download(
    url: str,
    dest: Path,
    progress: Progress = None,
    max_attempts: int = 500,
    stall_limit: int = 40,
    label: str = "",
) -> bool:
    """Fetch `url` to `dest`, resuming an existing partial file.

    Returns True only when the file on disk matches the server's
    Content-Length.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    name = label or dest.name

    try:
        total = content_length(url)
    except Exception as exc:
        _report(progress, f"Could not reach {url}: {exc}")
        return False

    if dest.exists() and dest.stat().st_size > total:
        _report(progress, f"{name}: existing file is oversized (corrupt); starting over")
        dest.unlink()

    stalled = 0
    for attempt in range(1, max_attempts + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if have == total:
            return True

        headers = dict(UA)
        headers["Range"] = f"bytes={have}-"
        before = have

        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=TIMEOUT
            ) as response:
                if have > 0 and not _range_honoured(response, have, progress, name):
                    dest.unlink(missing_ok=True)
                    continue
                with open(dest, "ab") as handle:
                    while True:
                        block = response.read(CHUNK)
                        if not block:
                            break
                        handle.write(block)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            _report(progress, f"{name}: {type(exc).__name__} - retrying")

        now = dest.stat().st_size if dest.exists() else 0
        if now > total:
            with open(dest, "r+b") as handle:
                handle.truncate(before)
            now = before

        if now == total:
            _report(progress, f"{name}: complete ({total:,} bytes)")
            return True

        _report(progress, f"{name}: {100.0 * now / total:.1f}% ({now:,}/{total:,})")

        if now <= before:
            stalled += 1
            if stalled >= stall_limit:
                _report(progress, f"{name}: no progress, giving up")
                return False
            time.sleep(3)
        else:
            stalled = 0

    return dest.exists() and dest.stat().st_size == total


def _range_honoured(response, offset: int, progress: Progress, name: str) -> bool:
    """A resumed request must come back as 206 starting at our exact offset."""
    if response.status != 206:
        _report(progress, f"{name}: server ignored Range (status {response.status}); restarting")
        return False
    header = response.headers.get("Content-Range", "")
    try:
        start = int(header.split()[-1].split("-")[0])
    except (ValueError, IndexError):
        _report(progress, f"{name}: unreadable Content-Range {header!r}; restarting")
        return False
    if start != offset:
        _report(progress, f"{name}: Content-Range starts at {start}, expected {offset}; restarting")
        return False
    return True


def download_and_extract_zip(url: str, dest_dir: Path, progress: Progress = None) -> bool:
    """Fetch a zip archive and unpack it, flattening a single top-level folder.

    GitHub zipballs wrap everything in a `<repo>-<sha>/` directory, which is
    stripped so callers get a stable path.
    """
    dest_dir = Path(dest_dir)
    archive = dest_dir.parent / f"{dest_dir.name}.zip"

    if not download(url, archive, progress=progress, label=f"{dest_dir.name} source"):
        return False

    try:
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            roots = {n.split("/")[0] for n in names if "/" in n}
            strip = roots.pop() + "/" if len(roots) == 1 else ""
            for member in names:
                if member.endswith("/"):
                    continue
                relative = member[len(strip):] if strip and member.startswith(strip) else member
                if not relative:
                    continue
                target = dest_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, open(target, "wb") as out:
                    out.write(src.read())
    except (zipfile.BadZipFile, OSError) as exc:
        _report(progress, f"Could not unpack {archive.name}: {exc}")
        archive.unlink(missing_ok=True)
        return False

    archive.unlink(missing_ok=True)
    return True
