from __future__ import annotations

from typing import Any


def json_safe(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]

    try:
        import numpy as np  # type: ignore

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    try:
        import pandas as pd  # type: ignore

        if isinstance(obj, pd.Timestamp):
            return obj.to_pydatetime().isoformat()
        if isinstance(obj, pd.Timedelta):
            return obj.to_pytimedelta().total_seconds()
        if isinstance(obj, pd.Series):
            return obj.to_dict()
    except Exception:
        pass

    try:
        return obj.item()  # type: ignore[attr-defined]
    except Exception:
        return str(obj)


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        results: list[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, str):
                results.append(item)
            else:
                results.append(str(json_safe(item)))
        return results
    if isinstance(value, str):
        return [value]
    return [str(json_safe(value))]


def coerce_responses(record: dict) -> list[str]:
    if "responses" in record:
        return _coerce_list(record.get("responses"))
    if "response" in record:
        return _coerce_list(record.get("response"))
    return []


def coerce_id(record: dict) -> str:
    if "id" in record and record.get("id") is not None:
        return str(json_safe(record.get("id")))
    if "sample_id" in record and record.get("sample_id") is not None:
        return str(json_safe(record.get("sample_id")))
    return ""
