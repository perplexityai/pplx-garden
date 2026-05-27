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