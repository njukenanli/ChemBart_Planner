"""Download ChemBart planner assets from Hugging Face.

This script downloads model checkpoints and the basic molecule dataset into the
locations expected by the planner configuration.

Usage examples:
    python CB_Planner/data/download.py
    python CB_Planner/data/download.py --only chembart rl
    python CB_Planner/data/download.py --force
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


ASSETS = {
    "chembart": {
        "url": "https://huggingface.co/ChemBart/Pretrained-Full/resolve/main/ChemBart_Full4.pth",
        "dest": Path("CB_Planner/functions/ChemBart/model/Pretrained-Full.pth"),
        "description": "Pretrained ChemBart model",
    },
    "rl": {
        "url": "https://huggingface.co/ChemBart/Policy-Value/resolve/main/CB_MCTS.pth",
        "dest": Path("CB_Planner/functions/ChemBart/model/Policy-Value.pth"),
        "description": "Policy-value model",
    },
    "temp_yield": {
        "url": "https://huggingface.co/ChemBart/Temperature-Yield/resolve/main/temp_yield_bart.pth",
        "dest": Path("CB_Planner/functions/ChemBart/model/Temperature-Yield.pth"),
        "description": "Temperature-yield model",
    },
    "basic_mol": {
        "url": "https://huggingface.co/datasets/ChemBart/Basic-Mols/resolve/main/basic_mol_canon.json",
        "dest": Path("CB_Planner/functions/ChemBart/data/basic_mol.json"),
        "description": "Buyable molecule dataset",
    },
}


def _download(url: str, destination: Path, force: bool, timeout: int = 60) -> bool:
    """Download one asset.

    Returns True when a file is downloaded, False when skipped.
    """
    if destination.exists() and not force:
        print(f"[skip] {destination} already exists")
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")

    try:
        with urlopen(url, timeout=timeout) as response, temp_path.open("wb") as fout:
            shutil.copyfileobj(response, fout)
    except (HTTPError, URLError, TimeoutError) as exc:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"failed to download {url}: {exc}") from exc

    temp_path.replace(destination)
    size_mb = destination.stat().st_size / (1024 * 1024)
    print(f"[ok] {destination} ({size_mb:.2f} MB)")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ChemBart planner assets")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(ASSETS.keys()),
        help="Only download selected assets.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Network timeout in seconds for each file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_keys = args.only or list(ASSETS.keys())

    print("Assets to download:")
    for key in selected_keys:
        meta = ASSETS[key]
        print(f"  - {key}: {meta['description']} -> {meta['dest']}")

    errors = []
    downloaded = 0

    for key in selected_keys:
        meta = ASSETS[key]
        try:
            if _download(meta["url"], meta["dest"], force=args.force, timeout=args.timeout):
                downloaded += 1
        except RuntimeError as exc:
            errors.append(str(exc))
            print(f"[error] {exc}")

    print(f"Completed. Downloaded {downloaded} file(s).")
    if errors:
        print("Some downloads failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
