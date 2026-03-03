import tempfile
from pathlib import Path
from urllib.parse import urlparse

from verl.eval.datasets.simpleqa import DEFAULT_URL, PROMPT_FALLBACK, SimpleQADataset


def _write_dummy_csv(cache_dir: Path) -> None:
    filename = Path(urlparse(DEFAULT_URL).path).name or "simpleqa.csv"
    csv_path = cache_dir / filename
    csv_path.write_text(
        "problem_id,problem,answer,metadata\n1,Question?,Answer,{}\n",
        encoding="utf-8",
    )


def test_prompt_fallback_contains_placeholders():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        _write_dummy_csv(cache_dir)
        dataset = SimpleQADataset(cache_dir=cache_dir)
    template = dataset._prompt_template
    if template is None:
        template = PROMPT_FALLBACK
    for token in ("{question}", "{target}", "{predicted_answer}"):
        assert token in template


def test_parse_judge_output_letters():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        _write_dummy_csv(cache_dir)
        dataset = SimpleQADataset(cache_dir=cache_dir)
        assert dataset.parse_judge_output("A") == "correct"
        assert dataset.parse_judge_output("C") == "not_attempted"
        assert dataset.parse_judge_output("B") == "incorrect"


def test_parse_judge_output_words():
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        _write_dummy_csv(cache_dir)
        dataset = SimpleQADataset(cache_dir=cache_dir)
        assert dataset.parse_judge_output("Correct") == "correct"
        assert dataset.parse_judge_output("Not_Attempted") == "not_attempted"
        assert dataset.parse_judge_output("Incorrect") == "incorrect"
