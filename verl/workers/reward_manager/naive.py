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

import inspect
from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


def _supports_reward_mode(fn) -> bool:
    """Return True if the callable accepts a `reward_mode` keyword."""
    try:
        sig = inspect.signature(inspect.unwrap(fn))
    except (ValueError, TypeError):
        return False
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return True
    return "reward_mode" in sig.parameters


@register("naive")
class NaiveRewardManager:
    """The reward manager."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        reward_mode="binary",
        **kwargs,
    ) -> None:
        """
        Initialize the NaiveRewardManager instance.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.reward_mode = reward_mode
        self._compute_score_supports_reward_mode = _supports_reward_mode(self.compute_score)

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        # First pass: decode and score all items.
        all_scores = []
        all_valid_response_lengths = []
        all_data_sources = []
        all_print_infos = []  # (prompt_str, response_str, ground_truth)

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            ability = data_item.non_tensor_batch.get("ability", None)
            if ability is not None:
                extra_info["ability"] = ability
            extra_info["num_turns"] = num_turns

            score_kwargs = dict(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            if self.reward_mode is not None and self._compute_score_supports_reward_mode:
                score_kwargs["reward_mode"] = self.reward_mode

            score = self.compute_score(**score_kwargs)

            all_scores.append(score)
            all_valid_response_lengths.append(valid_response_length)
            all_data_sources.append(data_source)
            all_print_infos.append((prompt_str, response_str, ground_truth))

        # Group-level adjustment for GRPO with ternary_adaptive reward mode:
        # if every rollout in a uid group fails to get the correct answer AND the
        # current rollout is a non-attempt (not a refusal ability sample), reward
        # the non-attempt with 0.5 instead of 0.0.
        # This prevents penalising honest uncertainty when the question is genuinely hard.
        uid_array = data.non_tensor_batch.get("uid", None)
        if self.reward_mode == "ternary_adaptive" and uid_array is not None:
            # Compute the best score seen per uid group.
            uid_to_max_score = {}
            for i, score in enumerate(all_scores):
                uid = uid_array[i]
                s = score["score"] if isinstance(score, dict) else float(score)
                if uid not in uid_to_max_score or s > uid_to_max_score[uid]:
                    uid_to_max_score[uid] = s

            # Track which uids have at least one adjusted rollout for logging.
            adjusted_uids = set()
            for i in range(len(data)):
                score = all_scores[i]
                if not isinstance(score, dict):
                    continue
                if not score.get("is_not_attempted", False):
                    continue
                uid = uid_array[i]
                ability = data[i].non_tensor_batch.get("ability", None)
                if ability is not None and ability.lower() == "refusal":
                    continue
                # Promote not-attempted reward from -1.0 → 0.0 when the whole group
                # got the answer wrong (no rollout reached score 1.0).
                if uid_to_max_score[uid] < 1.0:
                    all_scores[i] = dict(score)
                    all_scores[i]["score"] = 0
                    adjusted_uids.add(uid)

            # Log one group summary per adjusted uid for manual verification.
            uid_to_indices = defaultdict(list)
            for i, uid in enumerate(uid_array):
                uid_to_indices[uid].append(i)
            for uid in adjusted_uids:
                prompt_str = all_print_infos[uid_to_indices[uid][0]][0]
                print(f"[ternary_adaptive] uid={uid[:8]} prompt={prompt_str[:120]!r}")
                for idx in uid_to_indices[uid]:
                    s = all_scores[idx]
                    _, response_str, ground_truth = all_print_infos[idx]
                    ability = data[idx].non_tensor_batch.get("ability", None)
                    print(
                        f"  score={s['score']:.1f} not_attempted={s.get('is_not_attempted', False)}"
                        f" ability={ability} gt={str(ground_truth)[:40]!r} response={response_str[:80]!r}"
                    )

        # Second pass: fill reward tensor and handle printing.
        for i in range(len(data)):
            score = all_scores[i]
            valid_response_length = all_valid_response_lengths[i]
            data_source = all_data_sources[i]
            prompt_str, response_str, ground_truth = all_print_infos[i]

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
