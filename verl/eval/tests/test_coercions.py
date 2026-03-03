from verl.eval.io import coerce_id, coerce_responses


def test_coerce_responses_list():
    record = {"responses": ["a", "b"]}
    assert coerce_responses(record) == ["a", "b"]


def test_coerce_responses_single():
    record = {"response": "hello"}
    assert coerce_responses(record) == ["hello"]


def test_coerce_responses_missing():
    record = {"id": "1"}
    assert coerce_responses(record) == []


def test_coerce_id_priority():
    record = {"id": "abc", "sample_id": "def"}
    assert coerce_id(record) == "abc"


def test_coerce_id_fallback():
    record = {"sample_id": 123}
    assert coerce_id(record) == "123"


def test_coerce_id_missing():
    record = {"responses": ["x"]}
    assert coerce_id(record) == ""
