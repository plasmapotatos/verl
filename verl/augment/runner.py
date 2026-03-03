"""Runner for dataset augmentation."""

from __future__ import annotations

import hashlib
import os
from copy import deepcopy
from typing import Dict, Iterable, List, Optional

from datasets import Dataset
from tqdm import tqdm

from .io import read_parquet, write_parquet
from .registry import get as get_rewriter
from .schemas import attach_augmentation_metadata


def _derive_seed(global_seed: int, sample_id: str, method: str, variant_idx: int) -> int:
    payload = f"{global_seed}|{sample_id}|{method}|{variant_idx}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def run(
    input_path: str,
    output_path: Optional[str],
    output_dir: Optional[str],
    methods: List[str],
    n_variants_per_method: int,
    seed: int,
    write_per_method: bool,
    mix_original: bool,
    mode: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> None:
    dataset = read_parquet(input_path)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    samples = [dataset[i] for i in range(len(dataset))]

    outputs: List[dict] = []
    per_method_outputs: Dict[str, List[dict]] = {m: [] for m in methods}

    for idx, sample in enumerate(tqdm(samples, desc="Augmenting samples")):
        sample_id = None
        extra_info = sample.get("extra_info")
        if isinstance(extra_info, dict):
            sample_id = extra_info.get("sample_id")
        if sample_id is None:
            sample_id = str(idx)
        if mix_original:
            outputs.append(deepcopy(sample))
            for m in methods:
                per_method_outputs[m].append(deepcopy(sample))

        for method in methods:
            rewriter_kwargs = {"mode": mode} if mode is not None else {}
            rewriter = get_rewriter(method, **rewriter_kwargs)
            for variant_idx in range(n_variants_per_method):
                derived_seed = _derive_seed(seed, str(sample_id), method, variant_idx)
                augmented_samples = rewriter.rewrite(sample, rng_seed=derived_seed)
                for out in augmented_samples:
                    extra_info_out = out.get("extra_info", {}) if isinstance(out, dict) else {}
                    if not isinstance(extra_info_out, dict) or "augmentation" not in extra_info_out:
                        out = attach_augmentation_metadata(
                            out,
                            method_name=method,
                            variant_idx=variant_idx,
                            params={},
                            seed=derived_seed,
                        )
                    outputs.append(out)
                    per_method_outputs[method].append(out)

    if write_per_method:
        if output_dir is None:
            raise ValueError("output_dir is required when write_per_method is True")
        os.makedirs(output_dir, exist_ok=True)
        for method, items in per_method_outputs.items():
            path = os.path.join(output_dir, f"{method}.parquet")
            write_parquet(items, path)
    else:
        if output_path is None:
            raise ValueError("output_path is required when write_per_method is False")
        write_parquet(outputs, output_path)
