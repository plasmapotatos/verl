#!/usr/bin/env python3
"""Run a single prompt from input.txt and write response to output.txt."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Model path or HF repo id",
    )
    parser.add_argument("--input", default="input.txt", help="Input prompt file")
    parser.add_argument("--output", default="output.txt", help="Output response file")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    prompt = input_path.read_text(encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
        device_map="auto",
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    output_path.write_text(text, encoding="utf-8")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
