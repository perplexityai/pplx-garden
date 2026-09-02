use super::*;

#[test]
fn stop_token_at_the_generation_limit_still_counts_as_stopped() {
    assert!(ends_with_stop_token(&[10, 99], &[99, 100]));
    assert!(!ends_with_stop_token(&[10, 98], &[99, 100]));
    assert!(!ends_with_stop_token(&[], &[99, 100]));
}
