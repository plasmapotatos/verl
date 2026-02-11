"""Load a parquet file and print the first few records for validation."""

import argparse
import json
import os

import datasets
from PIL import Image

try:
    from verl.utils.dataset.vision_utils import process_image
except Exception:  # pragma: no cover - optional dependency
    process_image = None


def _to_jsonable(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return str(obj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_path", help="Path to a parquet file (train.parquet or test.parquet).")
    parser.add_argument(
        "--save-path",
        default="preview_parquet.json",
        help="Path to save preview rows as JSON. Default: preview_parquet.json.",
    )
    parser.add_argument("--num", type=int, default=5, help="Number of records to print.")
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Validate images field: check for empty lists or non-PIL images.",
    )
    args = parser.parse_args()

    parquet_path = os.path.expanduser(args.parquet_path)
    dataset = datasets.load_dataset("parquet", data_files=parquet_path)["train"]

    dataset_length = len(dataset)
    n = min(args.num, dataset_length)
    preview_rows = []
    for i in range(n):
        row = _to_jsonable(dataset[i])
        preview_rows.append({"index": i, "row": row})
        print(f"=== row {i} ===")
        print(json.dumps(row, ensure_ascii=False, indent=2))

    save_path = os.path.expanduser(args.save_path)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            {"dataset_length": dataset_length, "rows": preview_rows},
            f,
            ensure_ascii=False,
            indent=2,
        )

    if args.check_images:
        empty_count = 0
        non_pil_count = 0
        total_with_images = 0
        for i in range(len(dataset)):
            row = dataset[i]
            images = row.get("images")
            if images is None:
                empty_count += 1
                continue
            total_with_images += 1
            if len(images) == 0:
                empty_count += 1
                continue
            first = images[0]
            if isinstance(first, Image.Image):
                continue
            if process_image is not None:
                try:
                    processed = process_image(first)
                    if not isinstance(processed, Image.Image):
                        non_pil_count += 1
                except Exception:
                    non_pil_count += 1
            else:
                non_pil_count += 1

        print("=== image check summary ===")
        print(f"rows with images field: {total_with_images}")
        print(f"rows with empty images or image is None list: {empty_count}")
        print(f"rows with non-PIL images: {non_pil_count}")


if __name__ == "__main__":
    main()
