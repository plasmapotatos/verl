# VERL Augmentation

This module provides a lightweight plugin architecture for augmenting VERL-format parquet datasets.

## Usage

List available methods:

```
python -m verl.augment.cli --input /path/to/input.parquet --list_methods
```

Run augmentation using the built-in `templates` method:

```
python -m verl.augment.cli \
  --input /path/to/input.parquet \
  --output /path/to/output.parquet \
  --method templates \
  --n 1 \
  --seed 123 \
  --mix_original
```

Write per-method outputs:

```
python -m verl.augment.cli \
  --input /path/to/input.parquet \
  --output_dir /path/to/outputs \
  --method templates \
  --write_per_method
```

## Rewriter Interface

Implement a rewriter by registering a factory with `@register("method_name")` and exposing:

- `name: str`
- `rewrite(sample: dict, *, rng_seed: int | None = None) -> list[dict]`

Rewriters must preserve the schema. They should only modify `prompt` and may add
`extra_info["augmentation"]` metadata.

## LLM Rewriter (mode-based prompts)

The `llm_rewriter` method uses a mode-specific system prompt loaded from
`verl/augment/prompts/{mode}.txt`. The `mode` parameter selects the prompt file
and must exist, otherwise the rewriter raises an error.

Examples of modes:
- `paraphrase` uses `verl/augment/prompts/paraphrase.txt`
- `multisentence` would use `verl/augment/prompts/multisentence.txt`

The legacy method name `llm_paraphrase` is still accepted as an alias and uses
the same implementation with `mode="paraphrase"` by default.
