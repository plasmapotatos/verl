# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Bracket-aware SFT dataset.

Subclass of :class:`SFTDataset` that additionally emits a per-token
``bracket_mask`` marking prediction positions whose target token falls
strictly inside a ``[...]`` span in the response. Used by
:class:`BracketSubstitutionSFTTrainer` for online target substitution.

Backward compatible: only active when selected via ``data.custom_cls``.
"""

from __future__ import annotations

import re

import torch

from verl.utils.dataset.sft_dataset import SFTDataset


_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")


class BracketSFTDataset(SFTDataset):
    """SFTDataset that additionally emits a bracket-span mask.

    Convention: ``bracket_mask[i] == 1`` means "prediction at position i
    predicts a token that is strictly inside a bracket span". This matches
    the loss-mask convention in :class:`SFTDataset` (mask is keyed by
    prediction position, not by token position), so the trainer can use
    the same ``[:, :-1]`` slicing for both masks.
    """

    def __init__(self, parquet_files, tokenizer, config):
        super().__init__(parquet_files, tokenizer, config)
        bs_cfg = config.get("bracket_substitution", None) or {}
        # Whether to require strict containment (tokens whose char range
        # lies fully inside `[...]`). Tokens that merge a bracket with an
        # adjacent character (e.g. a single token "[P") are excluded.
        self._strict_containment = bool(bs_cfg.get("strict_containment", True))
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError(
                "BracketSFTDataset requires a fast tokenizer (return_offsets_mapping). "
                "Pass use_fast=True when loading the tokenizer."
            )

    def __getitem__(self, item):
        base = super().__getitem__(item)
        input_ids = base["input_ids"]
        loss_mask = base["loss_mask"]
        seq_len = input_ids.shape[0]

        bracket_mask = torch.zeros_like(loss_mask)

        response = self.responses[item]
        prompt = self.prompts[item]
        if not isinstance(response, str) or not isinstance(prompt, str):
            return {**base, "bracket_mask": bracket_mask}

        prompt_chat = [{"role": "user", "content": prompt}]
        prompt_chat_str = self.tokenizer.apply_chat_template(
            prompt_chat, add_generation_prompt=True, tokenize=False
        )
        response_chat_str = response + self.tokenizer.eos_token

        prompt_length = self.tokenizer(
            prompt_chat_str, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].shape[0]

        enc = self.tokenizer(
            response_chat_str,
            return_tensors="pt",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offsets = enc["offset_mapping"][0].tolist()

        # (char_start, char_end) of the CONTENT inside each `[...]`, exclusive of brackets.
        inside_spans = [(m.start() + 1, m.end() - 1) for m in _BRACKET_RE.finditer(response_chat_str)]
        if not inside_spans:
            bracket_mask = bracket_mask * loss_mask
            return {**base, "bracket_mask": bracket_mask}

        for tok_idx, (c_start, c_end) in enumerate(offsets):
            if c_end <= c_start:
                continue
            for (b_start, b_end) in inside_spans:
                if self._strict_containment:
                    in_span = c_start >= b_start and c_end <= b_end
                else:
                    in_span = c_end > b_start and c_start < b_end
                if in_span:
                    # Target token sits at input_ids[prompt_length + tok_idx].
                    # Prediction position is (target_pos - 1) by the same
                    # convention SFTDataset uses for loss_mask.
                    pred_idx = prompt_length + tok_idx - 1
                    if 0 <= pred_idx < seq_len:
                        bracket_mask[pred_idx] = 1
                    break

        bracket_mask = bracket_mask * loss_mask
        return {**base, "bracket_mask": bracket_mask}
