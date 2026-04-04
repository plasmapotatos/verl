"""Runner for dataset augmentation."""

from __future__ import annotations

import hashlib
import logging
import os
from copy import deepcopy
from typing import Dict, List, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from tqdm import tqdm

from .io import read_parquet, write_parquet
from .registry import get as get_rewriter
from .schemas import attach_augmentation_metadata


logger = logging.getLogger(__name__)

ResultType = tuple[List[dict], Dict[str, List[dict]]]


def _derive_seed(global_seed: int, sample_id: str, method: str, variant_idx: int) -> int:
    payload = f"{global_seed}|{sample_id}|{method}|{variant_idx}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _extract_sample_id(sample: dict, fallback: str) -> str:
    extra_info = sample.get("extra_info")
    if isinstance(extra_info, dict):
        sample_id = extra_info.get("sample_id")
        if sample_id is not None:
            return str(sample_id)
    sample_id = sample.get("sample_id")
    if sample_id is not None:
        return str(sample_id)
    sample_id = sample.get("id")
    if sample_id is not None:
        return str(sample_id)
    return fallback


def _collect_existing_keys(existing_path: str) -> set[tuple[str, str]]:
    existing_dataset = read_parquet(existing_path)
    keys: set[tuple[str, str]] = set()
    for idx in range(len(existing_dataset)):
        sample = existing_dataset[idx]
        extra_info = sample.get("extra_info")
        augmentation = extra_info.get("augmentation") if isinstance(extra_info, dict) else None
        if not isinstance(augmentation, dict):
            continue
        method = augmentation.get("method")
        if not isinstance(method, str) or not method:
            continue
        original_sample_id = augmentation.get("original_sample_id")
        if original_sample_id is None:
            original_sample_id = _extract_sample_id(sample, str(idx))
        keys.add((method, str(original_sample_id)))
    return keys


def run(
    input_path: str,
    output_path: Optional[str],
    output_dir: Optional[str],
    existing_path: Optional[str],
    methods: List[str],
    n_variants_per_method: int,
    seed: int,
    write_per_method: bool,
    mix_original: bool,
    mode: Optional[str] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 16,
) -> None:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if existing_path is not None and write_per_method:
        raise ValueError("existing_path is only supported when write_per_method is False")
    dataset = read_parquet(input_path)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    samples = [dataset[i] for i in range(len(dataset))]

    existing_outputs: List[dict] = []
    existing_keys: set[tuple[str, str]] = set()
    if existing_path is not None:
        existing_dataset = read_parquet(existing_path)
        existing_outputs = [existing_dataset[i] for i in range(len(existing_dataset))]
        existing_keys = _collect_existing_keys(existing_path)

    outputs: List[dict] = []
    per_method_outputs: Dict[str, List[dict]] = {m: [] for m in methods}
    rewriters = {}
    for method in methods:
        rewriter_kwargs = {"mode": mode} if mode is not None else {}
        rewriters[method] = get_rewriter(method, **rewriter_kwargs)

    def _process_sample(idx: int, sample: dict) -> tuple[List[dict], Dict[str, List[dict]]]:
        sample_id = _extract_sample_id(sample, str(idx))

        sample_outputs: List[dict] = []
        sample_per_method: Dict[str, List[dict]] = {m: [] for m in methods}
        if mix_original:
            sample_outputs.append(deepcopy(sample))
            for m in methods:
                sample_per_method[m].append(deepcopy(sample))

        for method in methods:
            rewriter = rewriters[method]
            if (method, sample_id) in existing_keys:
                logger.warning("Skipping already-processed sample_id=%s method=%s", sample_id, method)
                continue
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
                    sample_outputs.append(out)
                    sample_per_method[method].append(out)
        return sample_outputs, sample_per_method

    results: List[Optional[ResultType]] = [None] * len(samples)
    if batch_size <= 1:
        for idx, sample in enumerate(tqdm(samples, desc="Augmenting samples")):
            results[idx] = _process_sample(idx, sample)
    else:
        future_to_idx: Dict[Future[ResultType], int] = {}
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            for idx, sample in enumerate(samples):
                future = executor.submit(_process_sample, idx, sample)
                future_to_idx[future] = idx
            with tqdm(total=len(samples), desc="Augmenting samples") as progress:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    results[idx] = future.result()
                    progress.update(1)

    for idx, entry in enumerate(results):
        if entry is None:
            raise RuntimeError(f"Missing batch result for sample {idx}")
        sample_outputs, sample_per_method = entry
        outputs.extend(sample_outputs)
        for method, items in sample_per_method.items():
            per_method_outputs[method].extend(items)

    def _write_metrics(metrics: Dict[str, Dict[str, int]], path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            import json

            json.dump(metrics, handle, ensure_ascii=False, indent=2)

    metrics_by_method: Dict[str, Dict[str, int]] = {}
    for method, rewriter in rewriters.items():
        get_metrics = getattr(rewriter, "get_metrics", None)
        if callable(get_metrics):
            metrics = get_metrics()
            if isinstance(metrics, dict):
                metrics_by_method[method] = metrics

    if write_per_method:
        if output_dir is None:
            raise ValueError("output_dir is required when write_per_method is True")
        os.makedirs(output_dir, exist_ok=True)
        for method, items in per_method_outputs.items():
            path = os.path.join(output_dir, f"{method}.parquet")
            write_parquet(items, path)
            if method in metrics_by_method:
                metrics_path = os.path.join(output_dir, f"{method}_metrics.json")
                _write_metrics({method: metrics_by_method[method]}, metrics_path)
    else:
        if output_path is None and existing_path is None:
            raise ValueError("output_path is required when write_per_method is False")
        if existing_outputs:
            outputs = existing_outputs + outputs
        if output_path is None:
            output_path = existing_path
        write_parquet(outputs, output_path)
        if metrics_by_method:
            if output_path.endswith(".parquet"):
                metrics_path = output_path[: -len(".parquet")] + "_metrics.json"
            else:
                metrics_path = output_path + "_metrics.json"
            _write_metrics(metrics_by_method, metrics_path)
