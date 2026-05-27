"""Python bindings for the `pplx-unigram` Rust crate.

Loads a Hugging Face `tokenizer.json` and encodes text into token IDs via
Viterbi over a double-array trie. See `docs/unigram.md` for the underlying
implementation and `pplx-unigram/` for the Rust source.
"""

from pplx_garden._rust import UnigramEncodeState, UnigramEngine

Engine = UnigramEngine
EncodeState = UnigramEncodeState

__all__ = ["EncodeState", "Engine", "UnigramEncodeState", "UnigramEngine"]
