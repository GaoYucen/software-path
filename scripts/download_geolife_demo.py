#!/usr/bin/env python3
"""Download the small MovingPandas GeoLife demo files used for smoke tests."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


FILES = {
    "demodata_geolife.csv": (
        "https://raw.githubusercontent.com/movingpandas/movingpandas/main/"
        "tutorials/data/demodata_geolife.csv"
    ),
    "demodata_geolife.README": (
        "https://raw.githubusercontent.com/movingpandas/movingpandas/main/"
        "tutorials/data/demodata_geolife.README"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/raw")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in FILES.items():
        target = out_dir / filename
        print(f"Downloading {url} -> {target}")
        urlretrieve(url, target)


if __name__ == "__main__":
    main()
