//! Special-token matcher.

use crate::{Segment, TokenId};

#[derive(Debug, Clone)]
pub struct SpecialToken {
    pub id: TokenId,
    pub content: String,
    pub single_word: bool,
    pub lstrip: bool,
    pub rstrip: bool,
}

#[derive(Debug, Clone, Default)]
pub struct SpecialTokenMatcher {
    tokens: Vec<SpecialToken>,
}

impl SpecialTokenMatcher {
    pub fn from_specs(specs: Vec<SpecialToken>) -> Self {
        let mut tokens = specs;
        tokens.sort_by(|l, r| {
            r.content.len().cmp(&l.content.len()).then_with(|| l.id.cmp(&r.id))
        });
        Self { tokens }
    }

    pub fn split(
        &self,
        text: &str,
        segments: &mut Vec<Segment>,
        encode_special_tokens: bool,
    ) {
        segments.clear();
        if text.is_empty() {
            return;
        }
        if !encode_special_tokens || self.tokens.is_empty() {
            segments.push(Segment::text(0, text.len()));
            return;
        }

        let mut cursor = 0;
        while cursor < text.len() {
            // Pick the earliest match; on a tie, prefer the longest content
            // (so e.g. `</s>` beats `</`).
            let best = self
                .tokens
                .iter()
                .filter_map(|token| {
                    match_token(text, cursor, token)
                        .map(|(start, end)| (start, end, token))
                })
                .min_by(|a, b| {
                    a.0.cmp(&b.0).then_with(|| b.2.content.len().cmp(&a.2.content.len()))
                });
            let Some((start, end, token)) = best else {
                segments.push(Segment::text(cursor, text.len()));
                break;
            };
            if cursor < start {
                segments.push(Segment::text(cursor, start));
            }
            segments.push(Segment::special(start, end, token.id));
            cursor = end;
        }
    }
}

#[inline]
fn match_token(
    text: &str,
    cursor: usize,
    token: &SpecialToken,
) -> Option<(usize, usize)> {
    let relative_start = text[cursor..].find(token.content.as_str())?;
    let mut start = cursor + relative_start;
    let mut end = start + token.content.len();
    if token.single_word && !is_single_word_match(text, start, end) {
        return None;
    }
    if token.lstrip {
        start = trim_left_whitespace(text, start).max(cursor);
    }
    if token.rstrip {
        end = trim_right_whitespace(text, end);
    }
    Some((start, end))
}

#[inline]
fn is_single_word_match(text: &str, start: usize, end: usize) -> bool {
    let left =
        start == 0 || !text[..start].chars().next_back().is_some_and(is_word_character);
    let right =
        end == text.len() || !text[end..].chars().next().is_some_and(is_word_character);
    left && right
}

#[inline]
fn trim_left_whitespace(text: &str, mut start: usize) -> usize {
    while start > 0 {
        let Some(ch) = text[..start].chars().next_back() else { break };
        if !ch.is_whitespace() {
            break;
        }
        start -= ch.len_utf8();
    }
    start
}

#[inline]
fn trim_right_whitespace(text: &str, mut end: usize) -> usize {
    while end < text.len() {
        let Some(ch) = text[end..].chars().next() else { break };
        if !ch.is_whitespace() {
            break;
        }
        end += ch.len_utf8();
    }
    end
}

#[inline]
fn is_word_character(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '_'
}
