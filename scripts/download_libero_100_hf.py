#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download LIBERO-100 from Hugging Face by downloading:
- libero_10
- libero_90

Usage:
python scripts/download_libero_100_hf.py --download-dir /your/save/path
"""

import argparse
import os
from huggingface_hub import snapshot_download


HF_REPO_ID = "yifengzhu-hf/LIBERO-datasets"


def count_hdf5(folder: str) -> int:
    n = 0
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".hdf5"):
                n += 1
    return n


def download_libero_100(download_dir: str):
    os.makedirs(download_dir, exist_ok=True)

    # LIBERO-100 = LIBERO-10 + LIBERO-90
    subsets = ["libero_10", "libero_90"]

    for subset in subsets:
        print(f"[info] Downloading {subset} to {download_dir} ...")
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=download_dir,
            allow_patterns=f"{subset}/*",
        )
        subset_dir = os.path.join(download_dir, subset)
        num = count_hdf5(subset_dir) if os.path.exists(subset_dir) else 0
        print(f"[info] {subset} hdf5 files: {num}")

    n10 = count_hdf5(os.path.join(download_dir, "libero_10"))
    n90 = count_hdf5(os.path.join(download_dir, "libero_90"))
    print(f"[summary] libero_10={n10}, libero_90={n90}, total={n10+n90}")

    if n10 == 10 and n90 == 90:
        print("[ok] LIBERO-100 download complete.")
    else:
        print("[warn] File count is not expected (10 + 90). Please recheck network/download logs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download-dir",
        type=str,
        required=True,
        help="Folder to save datasets, e.g. /datasets/ljm_data/zyf/LIBERO/libero_data/official",
    )
    args = parser.parse_args()
    download_libero_100(args.download_dir)
