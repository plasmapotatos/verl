"""Extract sample_id list from a parquet file and save to JSON."""

import argparse
import json
import os

import datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_path", help="Path to a parquet file (train.parquet or test.parquet).")
    parser.add_argument(
        "--save-path",
        default="sample_ids.json",
        help="Path to save sample_id list as JSON. Default: sample_ids.json.",
    )
    args = parser.parse_args()

    parquet_path = os.path.expanduser(args.parquet_path)
    dataset = datasets.load_dataset("parquet", data_files=parquet_path)["train"]

    sample_ids = []
    for i in range(len(dataset)):
        row = dataset[i]
        sample_id = row.get("extra_info", {}).get("sample_id", row.get("sample_id"))
        if sample_id is not None:
            sample_ids.append(sample_id)

    save_path = os.path.expanduser(args.save_path)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(sample_ids, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(sample_ids)} sample_ids to {save_path}")


if __name__ == "__main__":
    main()
