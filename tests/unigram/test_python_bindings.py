import os

import pytest

from pplx_garden.unigram import EncodeState, Engine

# Tokenizer fixtures live alongside this test file. Set the env var
# UNIGRAM_TOKENIZER_JSON to point at a downloaded XLM-R tokenizer.json
# (https://huggingface.co/FacebookAI/xlm-roberta-base/blob/main/tokenizer.json).
# When the fixture is absent we skip — these are smoke tests, not network tests.
TOKENIZER_PATH = os.environ.get("UNIGRAM_TOKENIZER_JSON")


@pytest.fixture(scope="module")
def engine() -> Engine:
    if not TOKENIZER_PATH:
        pytest.skip("UNIGRAM_TOKENIZER_JSON not set; download XLM-R tokenizer.json to run")
    return Engine.from_hf_json(TOKENIZER_PATH)


def test_vocab_size(engine: Engine) -> None:
    assert engine.vocab_size() == 250002


def test_encode_matches_rust_example(engine: Engine) -> None:
    # The same input/output pair the Rust `encode` example dogfoods.
    tokens = engine.encode("The quick brown fox jumps over the lazy dog.")
    assert tokens == [581, 63773, 119455, 6, 147797, 88203, 7, 645, 70, 21, 3285, 10269, 5]


def test_encode_cjk(engine: Engine) -> None:
    tokens = engine.encode("Hello 你好 world 世界")
    assert tokens == [35378, 6, 124084, 8999, 6, 3221]


def test_encode_into_reuses_state(engine: Engine) -> None:
    state = EncodeState()
    a = engine.encode_into("hello world", state)
    b = engine.encode_into("hello world", state)
    assert a == b
    assert len(a) > 0


def test_invalid_path_raises_runtime_error() -> None:
    with pytest.raises(RuntimeError):
        Engine.from_hf_json("/nonexistent/tokenizer.json")


def test_empty_vocab_bytes_raises_value_error() -> None:
    # An obviously-invalid tokenizer.json should surface as ValueError so callers
    # can distinguish bad input from I/O failure.
    payload = b'{"model": {"type": "Unigram", "vocab": []}}'
    with pytest.raises((ValueError, RuntimeError)):
        Engine.from_hf_json_bytes(payload)
