//! Pre-tokenization behaviour: `WhitespaceSplit` composed with `Metaspace`.
//!
//! These build a tiny synthetic `tokenizer.json` so the whole pipeline runs
//! without downloading a real model.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use pplx_unigram::{EncodeState, Engine};

/// A `precompiled_charsmap` whose trie matches nothing, so the normalizer is
/// the identity. The blob layout is `[u32 trie byte length][trie][normalized]`;
/// 256 zeroed trie slots keep every `node_pos ^= byte` lookup in bounds while
/// no label ever matches.
fn identity_charsmap_b64() -> String {
    let mut blob = 1024u32.to_le_bytes().to_vec();
    blob.resize(4 + 1024, 0);
    STANDARD.encode(blob)
}

/// Scores are irrelevant to whitespace handling, so pieces are simply ranked
/// longest-first to make the expected segmentation obvious.
fn tokenizer_json(vocab: &[&str], add_prefix_space: bool, added: &str) -> String {
    let entries: Vec<String> = vocab
        .iter()
        .map(|piece| {
            let score = -(20.0 - piece.chars().count() as f64);
            format!("[{}, {score}]", serde_json::to_string(piece).unwrap())
        })
        .collect();
    format!(
        r#"{{
          "added_tokens": [{added}],
          "normalizer": {{"type": "Precompiled", "precompiled_charsmap": "{}"}},
          "pre_tokenizer": {{"type": "Sequence", "pretokenizers": [
              {{"type": "WhitespaceSplit"}},
              {{"type": "Metaspace", "replacement": "▁",
                "add_prefix_space": {add_prefix_space}}}
          ]}},
          "decoder": {{"type": "Metaspace", "replacement": "▁",
                       "add_prefix_space": {add_prefix_space}}},
          "model": {{"type": "Unigram", "unk_id": 0, "vocab": [{}]}}
        }}"#,
        identity_charsmap_b64(),
        entries.join(", "),
    )
}

const VOCAB: &[&str] = &[
    "<unk>",         // 0 — unk_id
    "\u{2581}",      // 1 — the lone metaspace piece a stray space would produce
    "\u{2581}hello", // 2
    "\u{2581}world", // 3
    "\u{2581}a",     // 4
    "\u{2581}b",     // 5
    "hello",         // 6 — bare forms, used by the add_prefix_space=false case
    "world",         // 7
];

const UNK: u32 = 0;
const SPACE: u32 = 1;
const HELLO: u32 = 2;
const WORLD: u32 = 3;
const A: u32 = 4;
const B: u32 = 5;

fn engine() -> Engine {
    Engine::from_hf_json_bytes(tokenizer_json(VOCAB, true, "").as_bytes()).unwrap()
}

fn encode(engine: &Engine, text: &str) -> Vec<u32> {
    let mut state = EncodeState::new();
    engine.encode(text, &mut state).unwrap();
    state.tokens.clone()
}

#[test]
fn trailing_whitespace_does_not_add_a_token() {
    let engine = engine();
    let expected = vec![HELLO];
    assert_eq!(encode(&engine, "hello"), expected);
    assert_eq!(encode(&engine, "hello "), expected, "single trailing space");
    assert_eq!(encode(&engine, "hello   "), expected, "repeated trailing space");
    assert_eq!(encode(&engine, "hello\t"), expected, "trailing tab");
    assert_eq!(encode(&engine, "hello\n"), expected, "trailing newline");
    assert_eq!(encode(&engine, "hello \t\n "), expected, "mixed trailing run");
}

#[test]
fn leading_whitespace_does_not_add_a_token() {
    let engine = engine();
    let expected = vec![HELLO];
    assert_eq!(encode(&engine, " hello"), expected);
    assert_eq!(encode(&engine, "   hello"), expected);
    assert_eq!(encode(&engine, "\thello"), expected);
    assert_eq!(encode(&engine, " \n hello"), expected);
    assert_eq!(encode(&engine, "  hello  "), expected, "leading and trailing");
}

#[test]
fn interior_whitespace_separates_words_exactly_once() {
    let engine = engine();
    let expected = vec![HELLO, WORLD];
    assert_eq!(encode(&engine, "hello world"), expected);
    assert_eq!(encode(&engine, "hello  world"), expected, "two spaces");
    assert_eq!(encode(&engine, "hello     world"), expected, "many spaces");
    assert_eq!(encode(&engine, "hello\tworld"), expected, "tab");
    assert_eq!(encode(&engine, "hello\n\nworld"), expected, "blank line");
    assert_eq!(encode(&engine, "  hello \t world  "), expected, "surrounded");
    assert_eq!(encode(&engine, "a b"), vec![A, B]);
}

#[test]
fn whitespace_only_input_encodes_to_nothing() {
    let engine = engine();
    assert_eq!(encode(&engine, " "), Vec::<u32>::new());
    assert_eq!(encode(&engine, "   "), Vec::<u32>::new());
    assert_eq!(encode(&engine, "\t"), Vec::<u32>::new());
    assert_eq!(encode(&engine, "\n\n"), Vec::<u32>::new());
    assert_eq!(encode(&engine, " \t\r\n "), Vec::<u32>::new());
    assert_eq!(encode(&engine, ""), Vec::<u32>::new());
}

/// The lone metaspace piece is reachable, but only as real content — never as
/// a by-product of a space that merely separated two words.
#[test]
fn lone_metaspace_piece_only_matches_literal_content() {
    let engine = engine();
    assert_eq!(encode(&engine, "\u{2581}"), vec![SPACE, SPACE]);
    assert!(!encode(&engine, "hello world ").contains(&SPACE));
}

/// Words are decoded independently, so a vocabulary entry that spans a word
/// boundary must never win — matching how HuggingFace feeds each split to the
/// Unigram model on its own.
#[test]
fn no_token_may_span_a_word_boundary() {
    let mut vocab = VOCAB.to_vec();
    vocab.push("\u{2581}hello\u{2581}world"); // id 8, the longest piece by far
    let json = tokenizer_json(&vocab, true, "");
    let engine = Engine::from_hf_json_bytes(json.as_bytes()).unwrap();
    assert_eq!(encode(&engine, "hello world"), vec![HELLO, WORLD]);
}

/// With `add_prefix_space = false` the splits carry no separator at all, so the
/// word boundary only survives because each word is decoded on its own.
#[test]
fn words_stay_separate_without_a_prefix_space() {
    let json = tokenizer_json(VOCAB, false, "");
    let engine = Engine::from_hf_json_bytes(json.as_bytes()).unwrap();
    assert_eq!(encode(&engine, "hello world"), vec![6, 7]);
    assert_eq!(encode(&engine, "  hello   world  "), vec![6, 7]);
}

/// Unknown characters fuse inside one word but not across a word boundary.
#[test]
fn unknown_characters_fuse_only_within_a_word() {
    let engine = engine();
    assert_eq!(encode(&engine, "zz"), vec![SPACE, UNK]);
    assert_eq!(encode(&engine, "z z"), vec![SPACE, UNK, SPACE, UNK]);
    assert_eq!(encode(&engine, "z z "), vec![SPACE, UNK, SPACE, UNK]);
}

/// The originally reported symptom: whitespace around a special token must not
/// leave a stray metaspace token behind.
#[test]
fn whitespace_around_special_tokens_is_dropped() {
    let added = r#"{"id": 9, "content": "<s>", "special": true},
                   {"id": 10, "content": "</s>", "special": true}"#;
    let json = tokenizer_json(VOCAB, true, added);
    let engine = Engine::from_hf_json_bytes(json.as_bytes()).unwrap();
    assert_eq!(encode(&engine, "<s>hello</s>"), vec![9, HELLO, 10]);
    assert_eq!(encode(&engine, "<s> hello </s>"), vec![9, HELLO, 10]);
    assert_eq!(encode(&engine, "  <s>  hello  </s>  "), vec![9, HELLO, 10]);
    assert_eq!(encode(&engine, "<s> </s>"), vec![9, 10]);
    assert_eq!(encode(&engine, "hello </s>"), vec![HELLO, 10]);
}
