# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
CLM dataset
- Each row contains a single text field.
- Supervision follows SFT convention using loss_mask (no labels field).
"""

from __future__ import annotations

from typing import List

import pandas as pd
import torch
from omegaconf.listconfig import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask


class CLMDataset(Dataset):
    """
    In-memory CLM dataset using a single text field.

    Arguments:
        config (OmegaConf): the data config
    """

    def __init__(self, parquet_files: str | ListConfig, tokenizer, config):
        text_key = config.get("text_key", None)
        max_length = config.get("max_length", 1024)
        truncation = config.get("truncation", "error")
        use_shm = config.get("use_shm", False)

        if not text_key:
            raise ValueError("data.text_key must be set for CLM dataset")

        assert truncation in ["error", "left", "right"]
        self.truncation = truncation
        self.use_shm = use_shm

        if not isinstance(parquet_files, ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.text_key = text_key
        self.max_length = max_length

        self._download()
        self._read_files()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)

    def _read_files(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, pandas.core.series.Series | numpy.ndarray) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        if self.text_key not in self.dataframe.columns:
            raise KeyError(f"text_key '{self.text_key}' not found in dataframe columns")

        texts = self.dataframe[self.text_key]
        try:
            texts = texts.apply(series_to_item)
        except Exception:
            print(f"texts={texts}")
            raise
        if isinstance(texts, pd.DataFrame):
            texts = texts.squeeze()
        self.texts = texts.tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        tokenizer = self.tokenizer
        text = self.texts[item]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("CLM dataset expects non-empty text strings")

        tokenized = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]

        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = (
                torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype)
                * self.tokenizer.pad_token_id
            )
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == "left":
                input_ids = input_ids[-self.max_length :]
                attention_mask = attention_mask[-self.max_length :]
            elif self.truncation == "right":
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
            elif self.truncation == "error":
                raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")
            else:
                raise NotImplementedError(f"Unknown truncation method {self.truncation}")

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        valid_tokens = int(loss_mask.sum().item())
        if valid_tokens > 0:
            last_idx = valid_tokens - 1
            loss_mask[last_idx] = 0

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "loss_mask": loss_mask,
        }
