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
import pandas as pd

from verl.utils import hf_tokenizer
from verl.utils.dataset.clm_dataset import CLMDataset


def test_clm_dataset(tmp_path):
    data = pd.DataFrame({"text": ["hello world", "this is a longer example"]})
    parquet_path = tmp_path / "clm.parquet"
    data.to_parquet(parquet_path)

    tokenizer = hf_tokenizer("deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct")
    config = {
        "text_key": "text",
        "max_length": 16,
        "truncation": "right",
    }

    dataset = CLMDataset(parquet_files=str(parquet_path), tokenizer=tokenizer, config=config)
    item = dataset[0]

    assert item["input_ids"].shape[0] == config["max_length"]
    assert item["attention_mask"].shape[0] == config["max_length"]
    assert item["loss_mask"].shape[0] == config["max_length"]

    pad_positions = item["attention_mask"] == 0
    assert pad_positions.sum().item() >= 0
    if pad_positions.any():
        assert item["loss_mask"][pad_positions].sum().item() == 0

    valid_tokens = int(item["attention_mask"].sum().item())
    if valid_tokens > 0:
        assert item["loss_mask"][valid_tokens - 1].item() == 0
