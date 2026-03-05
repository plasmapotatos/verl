"""Load a parquet file and print the first few records for validation."""

import argparse
import html
import json
import os

import datasets
from datasets.arrow_writer import SchemaInferenceError
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
        default=os.path.join("previews", "preview_parquet.json"),
        help="Path to save preview rows as JSON. Default: previews/preview_parquet.json.",
    )
    parser.add_argument(
        "--save-html",
        default=None,
        help="Path to save HTML preview. If omitted, HTML is not generated.",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=-1,
        help="Number of records to print (-1 for all).",
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Validate images field: check for empty lists or non-PIL images.",
    )
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    def _resolve_path(path):
        expanded = os.path.expanduser(path)
        if os.path.isabs(expanded):
            return expanded
        return os.path.join(repo_root, expanded)

    parquet_path = os.path.expanduser(args.parquet_path)
    save_path = _resolve_path(args.save_path)
    html_path = _resolve_path(args.save_html) if args.save_html else None

    try:
        dataset = datasets.load_dataset("parquet", data_files=parquet_path)["train"]
    except SchemaInferenceError:
        try:
            import pyarrow.parquet as pq  # type: ignore
        except Exception as exc:
            raise RuntimeError("pyarrow is required to inspect empty parquet files.") from exc

        pq_file = pq.ParquetFile(parquet_path)
        num_rows = pq_file.metadata.num_rows if pq_file.metadata is not None else 0
        if num_rows != 0:
            raise

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(
                {"dataset_length": 0, "rows": []},
                f,
                ensure_ascii=False,
                indent=2,
            )

        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        html_content = """
<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <title>Parquet Preview</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 16px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 6px; vertical-align: top; }
        th { position: sticky; top: 0; background: #f7f7f7; }
    </style>
</head>
<body>
    <p>Rows: 0</p>
    <table id=\"dataTable\">
        <thead>
            <tr>
                <th>index</th>
            </tr>
        </thead>
        <tbody></tbody>
    </table>
</body>
</html>
"""
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print("Dataset is empty (0 rows). Wrote empty preview files.")
        return

    dataset_length = len(dataset)
    n = dataset_length if args.num < 0 else min(args.num, dataset_length)
    preview_rows = []
    for i in range(n):
        row = _to_jsonable(dataset[i])
        preview_rows.append({"index": i, "row": row})

    print_n = min(5, dataset_length)
    for i in range(print_n):
        row = preview_rows[i]["row"]
        print(f"=== row {i} ===")
        print(json.dumps(row, ensure_ascii=False, indent=2))

    all_rows = preview_rows
    if html_path is not None:
        all_rows = []
        for i in range(n):
            row = _to_jsonable(dataset[i])
            all_rows.append({"index": i, "row": row})

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            {"dataset_length": dataset_length, "rows": preview_rows},
            f,
            ensure_ascii=False,
            indent=2,
        )

        if html_path is None:
            return

        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        headers = []
        header_set = set()
        for item in all_rows:
            for key in item["row"].keys():
                if key not in header_set:
                    header_set.add(key)
                    headers.append(key)

        def _cell_value(value):
                if isinstance(value, (dict, list)):
                        return json.dumps(value, ensure_ascii=False)
                return "" if value is None else str(value)

        rows_html = []
        for item in all_rows:
            row = item["row"]
            cells = []
            row_text_parts = [str(item["index"])]
            for key in headers:
                value = _cell_value(row.get(key))
                row_text_parts.append(value)
                cells.append(f"<td>{html.escape(value)}</td>")
            row_text = " ".join(row_text_parts).lower()
            rows_html.append(
                f"<tr data-row=\"{html.escape(row_text)}\">"
                f"<td>{item['index']}</td>"
                + "".join(cells)
                + "</tr>"
            )

        html_content = f"""
<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <title>Parquet Preview</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 16px; }}
        input, select {{ margin-right: 8px; padding: 6px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 6px; vertical-align: top; }}
        th {{ position: sticky; top: 0; background: #f7f7f7; }}
        tr.hidden {{ display: none; }}
    </style>
</head>
<body>
    <div>
        <label>Search:</label>
        <input id=\"searchInput\" type=\"text\" placeholder=\"type to filter...\" />
        <label>Column:</label>
        <select id=\"columnSelect\">
            <option value=\"__all__\">All</option>
            <option value=\"__index__\">index</option>
            {"".join([f"<option value=\"{html.escape(h)}\">{html.escape(h)}</option>" for h in headers])}
        </select>
    </div>
    <p>Rows: {dataset_length}</p>
    <table id=\"dataTable\">
        <thead>
            <tr>
                <th>index</th>
                {"".join([f"<th>{html.escape(h)}</th>" for h in headers])}
            </tr>
        </thead>
        <tbody>
            {"".join(rows_html)}
        </tbody>
    </table>
    <script>
        const searchInput = document.getElementById('searchInput');
        const columnSelect = document.getElementById('columnSelect');
        const table = document.getElementById('dataTable');

        function filterRows() {{
            const query = searchInput.value.toLowerCase();
            const column = columnSelect.value;
            const rows = table.querySelectorAll('tbody tr');

            rows.forEach(row => {{
                let haystack = '';
                if (column === '__all__') {{
                    haystack = row.getAttribute('data-row') || '';
                }} else {{
                    const headerCells = table.querySelectorAll('thead th');
                    let colIndex = 0;
                    headerCells.forEach((cell, idx) => {{
                        if (cell.textContent === column) colIndex = idx;
                    }});
                    const cell = row.children[colIndex];
                    haystack = cell ? cell.textContent.toLowerCase() : '';
                }}

                if (haystack.includes(query)) {{
                    row.classList.remove('hidden');
                }} else {{
                    row.classList.add('hidden');
                }}
            }});
        }}

        searchInput.addEventListener('input', filterRows);
        columnSelect.addEventListener('change', filterRows);
    </script>
</body>
</html>
"""

        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

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
