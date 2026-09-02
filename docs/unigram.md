# `pplx-unigram`

A unigram tokenizer encoder. Loads a HuggingFace `tokenizer.json` and encodes
text into token IDs via Viterbi over a double-array trie packed one node per
cache line.

Supports the SentencePiece-style unigram pipeline: precompiled charsmap
normalization → Metaspace pre-tokenization → Viterbi segmentation, plus
special-token splitting.

## Run the example

Get a unigram `tokenizer.json` (e.g. [XLM-R](https://huggingface.co/FacebookAI/xlm-roberta-base/blob/main/tokenizer.json)), then:

```
cargo run --release --example encode -p pplx-unigram -- \
    path/to/tokenizer.json "The quick brown fox jumps over the lazy dog."
```

## Use from Python

After `pip install .`, the tokenizer is available through `pplx_garden.unigram`:

```python
from pplx_garden.unigram import Engine

engine = Engine.from_hf_json("path/to/tokenizer.json")
print(engine.vocab_size())
print(engine.encode("The quick brown fox jumps over the lazy dog."))
```

For hot loops that encode many strings, reuse an `EncodeState`:

```python
from pplx_garden.unigram import EncodeState, Engine

engine = Engine.from_hf_json("path/to/tokenizer.json")
state = EncodeState()
for line in lines:
    tokens = engine.encode_into(line, state)
```

Loading is also supported from an in-memory `bytes` payload via
`Engine.from_hf_json_bytes(buf)` for callers that already hold the JSON.

`UnsupportedConfig` and `InvalidConfig` from the Rust crate surface as
`ValueError`; everything else surfaces as `RuntimeError`.