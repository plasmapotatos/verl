from verl.augment.rewriters.templates import TemplatesRewriter
from verl.augment.schemas import get_prompt_text


def _sample():
    return {
        "data_source": "simpleqa",
        "prompt": [{"role": "user", "content": "Hello?"}],
        "ability": "qa",
        "reward_model": {"style": "rule", "ground_truth": "Hi"},
        "extra_info": {"split": "train", "sample_id": "abc"},
    }


def test_rewriter_schema_preserving():
    sample = _sample()
    rewriter = TemplatesRewriter()
    outputs = rewriter.rewrite(sample, rng_seed=123)
    assert outputs
    for out in outputs:
        assert set(out.keys()) == set(sample.keys())
        assert get_prompt_text(out) != get_prompt_text(sample)
        assert "augmentation" in out.get("extra_info", {})
        assert set(out["extra_info"].keys()) == set(sample["extra_info"].keys()) | {"augmentation"}
