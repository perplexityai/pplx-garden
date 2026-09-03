use std::time::Duration;

use super::*;

fn completion(prompt: u64, cached: u64, completion: u64, ms: u64) -> RequestRecord {
    RequestRecord {
        method: "POST".to_string(),
        path: "/v1/chat/completions".to_string(),
        status: 200,
        elapsed: Duration::from_millis(ms),
        tokens: Some(TokenCounts { prompt, cached, completion }),
        error: None,
    }
}

fn failure(status: u16) -> RequestRecord {
    RequestRecord {
        method: "POST".to_string(),
        path: "/v1/chat/completions".to_string(),
        status,
        elapsed: Duration::from_millis(1),
        tokens: None,
        error: Some("boom".to_string()),
    }
}

#[test]
fn stats_bucket_requests_by_status_class() {
    let mut stats = Stats::new();
    stats.record(Event::Finished(completion(10, 0, 5, 100)));
    stats.record(Event::Finished(failure(400)));
    stats.record(Event::Finished(failure(500)));
    assert_eq!(stats.total, 3);
    assert_eq!(stats.ok, 1);
    assert_eq!(stats.client_errors, 1);
    assert_eq!(stats.server_errors, 1);
}

#[test]
fn stats_track_in_flight_request() {
    let mut stats = Stats::new();
    assert!(!stats.in_flight);
    stats.record(Event::Started);
    assert!(stats.in_flight);
    stats.record(Event::Finished(completion(10, 0, 5, 100)));
    assert!(!stats.in_flight);
}

#[test]
fn stats_cache_hit_ratio_covers_completions_only() {
    let mut stats = Stats::new();
    assert_eq!(stats.cache_hit_ratio(), None);
    stats.record(Event::Finished(completion(100, 50, 5, 100)));
    stats.record(Event::Finished(completion(100, 0, 5, 100)));
    stats.record(Event::Finished(failure(400)));
    assert_eq!(stats.cache_hit_ratio(), Some(0.25));
    assert_eq!(stats.prompt_tokens, 200);
    assert_eq!(stats.cached_tokens, 50);
    assert_eq!(stats.completion_tokens, 10);
}

#[test]
fn stats_rolling_rate_uses_last_ten_completions() {
    let mut stats = Stats::new();
    assert_eq!(stats.rolling_tok_per_s(), None);
    for _ in 0..2 {
        stats.record(Event::Finished(completion(1, 0, 1000, 1000)));
    }
    for _ in 0..10 {
        stats.record(Event::Finished(completion(1, 0, 10, 1000)));
    }
    // 100 tokens over 10 s; the two 1000-token records fall outside the window.
    assert_eq!(stats.rolling_tok_per_s(), Some(10.0));
}

#[test]
fn stats_keep_only_the_last_twenty_requests() {
    let mut stats = Stats::new();
    for i in 0..25u64 {
        stats.record(Event::Finished(completion(i, 0, 1, 1)));
    }
    assert_eq!(stats.recent.len(), 20);
    assert_eq!(stats.recent.front().unwrap().tokens.unwrap().prompt, 5);
    assert_eq!(stats.recent.back().unwrap().tokens.unwrap().prompt, 24);
}
