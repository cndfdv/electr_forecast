"""Dataset download helpers.

The downloader uses public source URLs and stores raw files under data/raw.
It does not overwrite existing files unless --force is passed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


URLS = {
    "etth1": (
        "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        "ETTh1.csv",
    ),
    "ecl": (
        "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/electricity/electricity.txt.gz",
        "electricity.txt.gz",
    ),
}


def download_dataset(dataset: str, output_dir: Path, force: bool = False) -> Path:
    """Download a raw dataset and return the local path."""
    if dataset not in URLS:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of {sorted(URLS)}.")

    url, filename = URLS[dataset]
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    if path.exists() and not force:
        return path

    urlretrieve(url, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(URLS), required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    path = download_dataset(args.dataset, args.output_dir, args.force)
    print(path)


if __name__ == "__main__":
    main()

