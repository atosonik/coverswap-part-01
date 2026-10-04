"""Split large files into small parts, and put them back together byte-exact.

Two part sets live in this project:

    parts/          model weights the app loads          (weights/)
    runableparts/   the built desktop app                  (dist/CoverSwap.exe)

    python chunker.py split                            # weights/ -> parts/
    python chunker.py merge                            # parts/ -> weights/

    python chunker.py split --src dist/CoverSwap.exe --parts runableparts
    python chunker.py verify --parts runableparts
    python chunker.py merge  --parts runableparts      # restores dist/CoverSwap.exe

Parts are capped at 1,000,000 bytes by default. Every part and every whole file
carries a SHA-256 in manifest.json; merge refuses corrupted parts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "weights")
PARTS = os.path.join(HERE, "parts")
MANIFEST = "manifest.json"
DEFAULT_CHUNK = 1_000_000
READ_BLOCK = 1 << 20

# Every file under weights/ that models.py checks before running.
REQUIRED = [
    "yolo11m-pose.pt",
    "sam2.1_b.pt",
    "sd15-inpaint-fp16/scheduler/scheduler_config.json",
    "sd15-inpaint-fp16/unet/config.json",
    "sd15-inpaint-fp16/unet/diffusion_pytorch_model.safetensors",
    "sd-vae-ft-mse/config.json",
    "sd-vae-ft-mse/diffusion_pytorch_model.safetensors",
    "catvton/mix-48k-1024/attention/model.safetensors",
    "faceswap/inswapper_128.onnx",
    os.path.join("faceswap", "insightface", "models", "buffalo_l", "det_10g.onnx"),
    os.path.join("faceswap", "insightface", "models", "buffalo_l", "w600k_r50.onnx"),
    os.path.join("faceswap", "insightface", "models", "buffalo_l", "2d106det.onnx"),
]


class ChunkError(Exception):
    """A part is missing, the wrong size, or fails its checksum."""


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000.0
    return f"{n} B"


def parse_size(text: str) -> int:
    t = text.strip().upper().replace(" ", "")
    for suffix, mult in (
        ("KIB", 1 << 10),
        ("MIB", 1 << 20),
        ("GIB", 1 << 30),
        ("KB", 1000),
        ("MB", 1000_000),
        ("GB", 1000_000_000),
        ("K", 1000),
        ("M", 1000_000),
        ("G", 1000_000_000),
    ):
        if t.endswith(suffix):
            return int(float(t[: -len(suffix)]) * mult)
    return int(t)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(READ_BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def resolve_sources(src: str, all_files: bool) -> tuple[str, list[str]]:
    src = os.path.abspath(src)
    if os.path.isfile(src):
        return os.path.dirname(src), [os.path.basename(src)]
    if not os.path.isdir(src):
        raise ChunkError(f"No such file or directory: {src}")

    is_weights = os.path.normcase(src) == os.path.normcase(os.path.abspath(WEIGHTS))
    if is_weights and not all_files:
        return src, [r for r in REQUIRED if os.path.exists(os.path.join(src, r))]

    found = []
    for root, _, names in os.walk(src):
        for n in sorted(names):
            found.append(os.path.relpath(os.path.join(root, n), src))
    return src, sorted(found)


def split_file(root: str, rel: str, parts_root: str, part_dir: str, chunk: int) -> dict:
    src = os.path.join(root, rel)
    out_dir = os.path.join(parts_root, part_dir) if part_dir else parts_root
    if part_dir and os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    total = os.path.getsize(src)
    count = max(1, -(-total // chunk))
    width = max(5, len(str(max(0, count - 1))))
    whole = hashlib.sha256()
    parts = []

    with open(src, "rb") as fh:
        for index in range(count):
            name = f"{index:0{width}d}.bin"
            part_hash = hashlib.sha256()
            written = 0
            with open(os.path.join(out_dir, name), "wb") as out:
                while written < chunk:
                    block = fh.read(min(READ_BLOCK, chunk - written))
                    if not block:
                        break
                    out.write(block)
                    part_hash.update(block)
                    whole.update(block)
                    written += len(block)
            if written == 0:
                os.remove(os.path.join(out_dir, name))
                break
            parts.append(
                {"file": name, "size": written, "sha256": part_hash.hexdigest()}
            )

    return {
        "path": rel.replace(os.sep, "/"),
        "parts_dir": part_dir.replace(os.sep, "/"),
        "size": total,
        "sha256": whole.hexdigest(),
        "parts": parts,
    }


def cmd_split(args) -> int:
    chunk = parse_size(args.size)
    if chunk < 1024:
        print("Chunk size must be at least 1 KB.", file=sys.stderr)
        return 2

    root, files = resolve_sources(args.src, args.all)
    if not files:
        print(f"Nothing to split under {root}.", file=sys.stderr)
        return 1

    parts_root = os.path.abspath(args.parts)
    os.makedirs(parts_root, exist_ok=True)
    flat = len(files) == 1 and os.path.isfile(os.path.abspath(args.src))

    entries, n_parts = [], 0
    for rel in files:
        part_dir = "" if flat else rel.replace("/", os.sep).replace("\\", os.sep) + ".parts"
        entry = split_file(root, rel, parts_root, part_dir, chunk)
        entries.append(entry)
        n_parts += len(entry["parts"])
        print(f"  {rel}  {human(entry['size'])} -> {len(entry['parts'])} parts")

    manifest = {
        "version": 2,
        "chunk_size": chunk,
        "root": os.path.relpath(root, HERE).replace(os.sep, "/"),
        "files": entries,
    }
    with open(os.path.join(parts_root, MANIFEST), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    total = sum(e["size"] for e in entries)
    print(
        f"\n{len(entries)} file(s), {n_parts} parts, {human(total)} in "
        f"{os.path.relpath(parts_root, HERE)}/"
    )
    return 0


def load_manifest(parts_root: str) -> dict:
    path = os.path.join(parts_root, MANIFEST)
    if not os.path.exists(path):
        raise ChunkError(f"No manifest at {path}. Run chunker.py split first.")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def entry_part_dir(parts_root: str, entry: dict) -> str:
    sub = entry.get("parts_dir", "")
    return os.path.join(parts_root, sub.replace("/", os.sep)) if sub else parts_root


def merge_file(entry: dict, parts_root: str, dest_root: str, force: bool) -> str:
    rel = entry["path"].replace("/", os.sep)
    dest = os.path.join(dest_root, rel)
    part_dir = entry_part_dir(parts_root, entry)

    if os.path.exists(dest) and not force:
        if os.path.getsize(dest) == entry["size"] and sha256_file(dest) == entry["sha256"]:
            return "skipped"

    missing = [
        p["file"]
        for p in entry["parts"]
        if not os.path.exists(os.path.join(part_dir, p["file"]))
    ]
    if missing:
        raise ChunkError(f"{rel}: {len(missing)} part(s) missing, first {missing[0]}")

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    tmp = dest + ".merging"
    whole = hashlib.sha256()
    try:
        with open(tmp, "wb") as out:
            for p in entry["parts"]:
                with open(os.path.join(part_dir, p["file"]), "rb") as fh:
                    data = fh.read()
                if len(data) != p["size"]:
                    raise ChunkError(
                        f"{rel}: part {p['file']} is {len(data)} bytes, expected {p['size']}"
                    )
                if hashlib.sha256(data).hexdigest() != p["sha256"]:
                    raise ChunkError(f"{rel}: part {p['file']} failed its checksum")
                out.write(data)
                whole.update(data)
        if whole.hexdigest() != entry["sha256"]:
            raise ChunkError(f"{rel}: assembled file failed its checksum")
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

    os.replace(tmp, dest)
    return "written"


def cmd_merge(args) -> int:
    parts_root = os.path.abspath(args.parts)
    manifest = load_manifest(parts_root)
    dest_root = (
        os.path.abspath(args.dest)
        if args.dest
        else os.path.abspath(os.path.join(HERE, manifest.get("root", "weights")))
    )

    written = skipped = 0
    for entry in manifest["files"]:
        result = merge_file(entry, parts_root, dest_root, args.force)
        written += result == "written"
        skipped += result == "skipped"
        print(f"  [{result}] {entry['path']}  {human(entry['size'])}")
    print(
        f"\n{written} written, {skipped} already correct, "
        f"into {os.path.relpath(dest_root, HERE)}/"
    )
    return 0


def cmd_verify(args) -> int:
    parts_root = os.path.abspath(args.parts)
    manifest = load_manifest(parts_root)
    bad = 0
    for entry in manifest["files"]:
        part_dir = entry_part_dir(parts_root, entry)
        problems = []
        for p in entry["parts"]:
            src = os.path.join(part_dir, p["file"])
            if not os.path.exists(src):
                problems.append(f"{p['file']} missing")
            elif os.path.getsize(src) != p["size"]:
                problems.append(f"{p['file']} wrong size")
            elif sha256_file(src) != p["sha256"]:
                problems.append(f"{p['file']} bad checksum")
        bad += len(problems)
        status = "ok" if not problems else f"{len(problems)} PROBLEM(S)"
        print(f"  [{status}] {entry['path']}  {len(entry['parts'])} parts")
        for problem in problems[:5]:
            print(f"       {problem}")
    print("\nAll parts verified." if not bad else f"\n{bad} bad part(s).")
    return 1 if bad else 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("split", help="cut a file or folder into parts")
    s.add_argument("--src", default=WEIGHTS, help="file or folder (default: weights/)")
    s.add_argument("--parts", default=PARTS, help="output folder (default: parts/)")
    s.add_argument("--size", default=str(DEFAULT_CHUNK), help="max part size (default 1000000)")
    s.add_argument("--all", action="store_true", help="split every file under weights/")
    s.set_defaults(func=cmd_split)

    m = sub.add_parser("merge", help="rebuild the original from parts")
    m.add_argument("--parts", default=PARTS, help="parts folder (default: parts/)")
    m.add_argument("--dest", default=None, help="restore destination (default: manifest root)")
    m.add_argument("--force", action="store_true", help="rebuild even if already correct")
    m.set_defaults(func=cmd_merge)

    v = sub.add_parser("verify", help="checksum every part without writing")
    v.add_argument("--parts", default=PARTS, help="parts folder (default: parts/)")
    v.set_defaults(func=cmd_verify)

    args = p.parse_args()
    try:
        return args.func(args)
    except ChunkError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        print("Re-split from a good copy, or re-fetch the damaged part.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
