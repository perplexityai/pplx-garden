use super::strict_prefix;

#[test]
fn cache_reuses_only_strict_token_prefixes() {
    assert!(strict_prefix(&[1, 2], &[1, 2, 3]));
    assert!(!strict_prefix(&[1, 2], &[1, 2]));
    assert!(!strict_prefix(&[1, 9], &[1, 2, 3]));
}
