"""Simple disk cache interface."""

from __future__ import annotations

import json
import os
from hashlib import md5
from typing import Any, Optional


class DiskCache:
    def __init__(self, base_dir: str) -> None:
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def _path_for_key(self, key: str) -> str:
        digest = md5(key.encode("utf-8")).hexdigest()
        return os.path.join(self.base_dir, f"{digest}.json")

    def get(self, key: str) -> Optional[Any]:
        path = self._path_for_key(key)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def set(self, key: str, value: Any) -> None:
        path = self._path_for_key(key)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
