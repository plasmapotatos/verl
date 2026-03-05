from __future__ import annotations

import csv
from pathlib import Path


def add_problem_id(input_csv: Path, output_csv: Path) -> None:
    with input_csv.open("r", encoding="utf-8", newline="") as f_in:
        reader = csv.reader(f_in)
        header = next(reader, None)
        if header is None:
            raise ValueError("Input CSV is empty.")

        new_header = header + ["id"]
        rows = list(reader)

    with output_csv.open("w", encoding="utf-8", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(new_header)
        for idx, row in enumerate(rows):
            writer.writerow(row + [str(idx)])


def main() -> None:
    input_csv = Path("/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/simple_qa_test_set.csv")
    output_csv = Path("/work/hdd/bbsg/twei2/rl/verl/data/simpleqa/simple_qa_test_set_with_ids.csv")
    add_problem_id(input_csv, output_csv)


if __name__ == "__main__":
    main()
