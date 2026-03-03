from verl.augment.schemas import get_prompt_text, set_prompt_text


def _sample():
    return {
        "data_source": "simpleqa",
        "prompt": [{"role": "user", "content": "Hello?"}],
        "ability": "qa",
        "reward_model": {"style": "rule", "ground_truth": "Hi"},
        "extra_info": {"split": "train", "sample_id": "abc"},
    }


def test_set_prompt_text_updates_content():
    sample = _sample()
    updated = set_prompt_text(sample, "Changed")
    assert get_prompt_text(updated) == "Changed"
    assert get_prompt_text(sample) == "Hello?"
