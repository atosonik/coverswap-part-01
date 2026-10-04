"""Fetch a small local library of family photos and garments for offline tests.

These are public Unsplash/Pexels photos plus the CatVTON demo garments already
on disk. After this script finishes, CoverSwap can be tried without hunting for
pictures.

    .venv\\Scripts\\python.exe scripts\\download_samples.py
"""

from __future__ import annotations

import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAMILY_DIR = ROOT / "samples" / "family"
CLOTHES_DIR = ROOT / "samples" / "clothes"
FORMAL_DIR = CLOTHES_DIR / "formal"
MILITARY_DIR = CLOTHES_DIR / "military"

# URLs below were spot-checked — Unsplash/Pexels IDs can drift, so filenames
# are descriptive and bad files can be deleted then re-run.
FAMILY = [
    (
        "family_park.jpg",
        "https://images.unsplash.com/photo-1511895426328-dc8714191300?auto=format&fit=crop&w=1400&q=80",
    ),
    (
        "family_sofa.jpg",
        "https://images.unsplash.com/photo-1609220136736-443140cffec6?auto=format&fit=crop&w=1400&q=80",
    ),
    (
        "family_kids.jpg",
        "https://images.unsplash.com/photo-1475503572774-15a45e5d60b9?auto=format&fit=crop&w=1400&q=80",
    ),
    (
        "family_walk.jpg",
        "https://images.unsplash.com/photo-1542037104857-ffbb0b9155fb?auto=format&fit=crop&w=1400&q=80",
    ),
    (
        "family_generations.jpg",
        "https://images.unsplash.com/photo-1766973116491-f13b0c8e1010?auto=format&fit=crop&w=1400&q=80",
    ),
    (
        "family_friends.jpg",
        "https://images.unsplash.com/photo-1529156069898-49953e39b3ac?auto=format&fit=crop&w=1400&q=80",
    ),
    (
        "family_picnic.jpg",
        "https://images.pexels.com/photos/4260325/pexels-photo-4260325.jpeg?auto=compress&cs=tinysrgb&w=1400",
    ),
    (
        "family_bed.jpg",
        "https://images.pexels.com/photos/4545205/pexels-photo-4545205.jpeg?auto=compress&cs=tinysrgb&w=1400",
    ),
]

CLOTHES = [
    (
        "dress_red.jpg",
        "https://images.unsplash.com/photo-1595777457583-95e059d581b8?auto=format&fit=crop&w=900&q=80",
    ),
    (
        "dress_pink.jpg",
        "https://images.unsplash.com/photo-1566174053879-31528523f8ae?auto=format&fit=crop&w=900&q=80",
    ),
    (
        "dress_white.jpg",
        "https://images.unsplash.com/photo-1515372039744-b8f02a3ae446?auto=format&fit=crop&w=900&q=80",
    ),
    (
        "shirt_black.jpg",
        "https://images.unsplash.com/photo-1583743814966-8936f5b7be1a?auto=format&fit=crop&w=900&q=80",
    ),
    (
        "shirt_white.jpg",
        "https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?auto=format&fit=crop&w=900&q=80",
    ),
    (
        "sweater_cream.jpg",
        "https://images.unsplash.com/photo-1434389677669-e08b4cac3105?auto=format&fit=crop&w=900&q=80",
    ),
]

FORMAL = [
    (
        "gown_wedding.jpg",
        "https://images.unsplash.com/photo-1718871743197-aee8056c508d?auto=format&fit=crop&w=900&q=80",
    ),
    (
        "gown_evening.jpg",
        "https://images.pexels.com/photos/265722/pexels-photo-265722.jpeg?auto=compress&cs=tinysrgb&w=900",
    ),
    (
        "dress_elegant.jpg",
        "https://images.unsplash.com/photo-1596783074918-c84cb06531ca?auto=format&fit=crop&w=900&q=80",
    ),
    (
        "suit_navy.jpg",
        "https://images.unsplash.com/photo-1594938298603-c8148c4dae35?auto=format&fit=crop&w=900&q=80",
    ),
    (
        "suit_black_hanger.jpg",
        "https://images.unsplash.com/photo-1585412459272-762fb93357c3?auto=format&fit=crop&w=900&q=80",
    ),
    (
        "blazer_black.jpg",
        "https://images.unsplash.com/photo-1603524961821-fcbbee4f0ae0?auto=format&fit=crop&w=900&q=80",
    ),
    (
        "tuxedo_flatlay.jpg",
        "https://images.unsplash.com/photo-1593030761757-71fae45fa0e7?auto=format&fit=crop&w=900&q=80",
    ),
    (
        "coat_formal.jpg",
        "https://images.unsplash.com/photo-1732165289122-713c00969877?auto=format&fit=crop&w=900&q=80",
    ),
]

MILITARY = [
    (
        "navy_dress_medals.jpg",
        "https://images.unsplash.com/photo-1714272663284-5ce8ab649ecd?auto=format&fit=crop&w=900&q=80",
    ),
    (
        "airforce_parade.jpg",
        "https://images.unsplash.com/photo-1751785405273-840a919548e3?auto=format&fit=crop&w=900&q=80",
    ),
    (
        "ceremonial_salute.jpg",
        "https://images.unsplash.com/photo-1770240366283-629dfa6be794?auto=format&fit=crop&w=900&q=80",
    ),
    (
        "jacket_patches.jpg",
        "https://images.unsplash.com/photo-1623792165505-b42b0c54372b?auto=format&fit=crop&w=900&q=80",
    ),
]

CATVTON = ROOT / "third_party" / "CatVTON" / "resource" / "demo" / "example"
LOCAL_COPIES = [
    (CATVTON / "condition" / "upper" / "24083449_54173465_2048.jpg", CLOTHES_DIR / "tee_rhude.jpg"),
    (CATVTON / "condition" / "upper" / "22790049_53294275_1000.jpg", CLOTHES_DIR / "vest_snoopy.jpg"),
    (CATVTON / "condition" / "upper" / "23255574_53383833_1000.jpg", CLOTHES_DIR / "tee_stripe.jpg"),
    (CATVTON / "condition" / "overall" / "21744571_51588794_1000.jpg", CLOTHES_DIR / "dress_blue.jpg"),
    (CATVTON / "condition" / "overall" / "23962182_54027982_1000.jpg", CLOTHES_DIR / "dress_floral.jpg"),
    (ROOT / "samples" / "bus.jpg", FAMILY_DIR / "street_group.jpg"),
]

UA = {"User-Agent": "Mozilla/5.0 (CoverSwap sample fetch)"}


def fetch(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 20_000:
        print(f"  already have {dest.name}", flush=True)
        return True
    try:
        request = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(request, timeout=45) as response:
            data = response.read()
        if len(data) < 8_000:
            print(f"  too small {dest.name} ({len(data)} bytes)", flush=True)
            return False
        dest.write_bytes(data)
        print(f"  saved {dest.name} ({len(data) / 1024:.0f} KB)", flush=True)
        return True
    except Exception as exc:
        print(f"  failed {dest.name}: {exc}", flush=True)
        dest.unlink(missing_ok=True)
        return False


def main() -> int:
    FAMILY_DIR.mkdir(parents=True, exist_ok=True)
    CLOTHES_DIR.mkdir(parents=True, exist_ok=True)
    FORMAL_DIR.mkdir(parents=True, exist_ok=True)
    MILITARY_DIR.mkdir(parents=True, exist_ok=True)

    print("Family photos", flush=True)
    family_ok = sum(fetch(url, FAMILY_DIR / name) for name, url in FAMILY)
    print("Everyday clothes", flush=True)
    clothes_ok = sum(fetch(url, CLOTHES_DIR / name) for name, url in CLOTHES)
    print("Formal clothes", flush=True)
    formal_ok = sum(fetch(url, FORMAL_DIR / name) for name, url in FORMAL)
    print("Military formals", flush=True)
    military_ok = sum(fetch(url, MILITARY_DIR / name) for name, url in MILITARY)

    print("Copying demo garments already on disk", flush=True)
    for src, dest in LOCAL_COPIES:
        if src.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            print(f"  copied {dest.name}", flush=True)
        else:
            print(f"  skip missing {src}", flush=True)

    families = list(FAMILY_DIR.glob("*.*"))
    clothes = [p for p in CLOTHES_DIR.rglob("*.*") if p.is_file()]
    print(
        f"Ready: {len(families)} family photos, {len(clothes)} clothes photos "
        f"({formal_ok} formal, {military_ok} military).",
        flush=True,
    )
    print(f"  {FAMILY_DIR}")
    print(f"  {CLOTHES_DIR}")
    if family_ok == 0 and not families:
        return 1
    if clothes_ok == 0 and formal_ok == 0 and military_ok == 0 and not clothes:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
