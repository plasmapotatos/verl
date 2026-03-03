from verl.augment.rewriters.llm_paraphrase import (
    _extract_question_segment,
    _replace_question_segment,
)


def test_extract_question_segment():
    prompt = "Question: What is 2+2?\nAnswer briefly."
    extracted = _extract_question_segment(prompt)
    assert extracted is not None
    prefix, question, suffix = extracted
    assert prefix == "Question: "
    assert question == "What is 2+2?"
    assert suffix == "\nAnswer briefly."


def test_replace_question_segment_preserves_suffix():
    prompt = "Question: What is 2+2?\nAnswer briefly."
    updated = _replace_question_segment(prompt, "Compute 2+2.")
    assert updated == "Question: Compute 2+2.\nAnswer briefly."


def test_replace_question_segment_no_match():
    prompt = "Just answer the question."
    updated = _replace_question_segment(prompt, "Provide a response.")
    assert updated == "Provide a response."
