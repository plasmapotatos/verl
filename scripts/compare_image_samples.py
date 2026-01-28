"""Compare image samples between ScienceQA hard-negative pairing and Geo3k parquet files."""

import argparse
import os

import datasets
from PIL import Image

from verl.utils.dataset.vision_utils import process_image


def _describe_image(img: Image.Image) -> dict:
    return {
        "type": type(img).__name__,
        "mode": img.mode,
        "size": img.size,
    }


def _find_first_with_images(dataset):
    for i in range(len(dataset)):
        row = dataset[i]
        images = row.get("images")
        if images:
            return i, row
    return None, None


def _load_dataset(path: str):
    path = os.path.expanduser(path)
    return datasets.load_dataset("parquet", data_files=path)["train"]


def _inspect_images(label: str, row: dict, output_dir: str | None):
    images = row.get("images")
    print(f"{label}: images field type={type(images).__name__}, length={len(images) if images is not None else 0}")
    if not images:
        return
    first = images[0]
    print(f"{label}: first image element type={type(first).__name__}")
    try:
        processed = process_image(first)
        print(f"{label}: processed image -> {_describe_image(processed)}")
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"{label.lower()}_sample.png")
            processed.save(out_path)
            print(f"{label}: saved processed image to {out_path}")
    except Exception as exc:
        print(f"{label}: process_image failed: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scienceqa_parquet",
        required=False,
        default="/work/hdd/bbsg/twei2/rl/verl/data/scienceqa_hard_negative_pairing/train.parquet",
        help="Path to ScienceQA parquet file (train.parquet or test.parquet).",
    )
    parser.add_argument(
        "--geo3k_parquet",
        required=False,
        default="/work/hdd/bbsg/twei2/rl/verl/data/geo3k/train.parquet",
        help="Path to Geo3k parquet file (train.parquet or test.parquet).",
    )
    parser.add_argument(
        "--output_dir",
        default="./image_debug",
        help="Directory to save processed images for visual inspection.",
    )
    args = parser.parse_args()

    scienceqa_ds = _load_dataset(args.scienceqa_parquet)
    geo3k_ds = _load_dataset(args.geo3k_parquet)

    s_idx, s_row = _find_first_with_images(scienceqa_ds)
    g_idx, g_row = _find_first_with_images(geo3k_ds)

    if s_row is None:
        print("ScienceQA: no rows with images found.")
    else:
        print(f"ScienceQA: first image at row {s_idx}")
        _inspect_images("ScienceQA", s_row, args.output_dir)

    if g_row is None:
        print("Geo3k: no rows with images found.")
    else:
        print(f"Geo3k: first image at row {g_idx}")
        _inspect_images("Geo3k", g_row, args.output_dir)


if __name__ == "__main__":
    main()
