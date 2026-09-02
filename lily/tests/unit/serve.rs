use super::*;

#[test]
fn request_surface_refuses_sampling_and_tools() {
    let sampling = br#"{
            "model":"Qwen3.6-35B-A3B",
            "messages":[{"role":"user","content":"hi"}],
            "temperature":0.7
        }"#;
    assert!(serde_json::from_slice::<ChatRequest>(sampling).is_err());

    let tools = br#"{
            "model":"Qwen3.6-35B-A3B",
            "messages":[{"role":"user","content":"hi"}],
            "tools":[]
        }"#;
    assert!(serde_json::from_slice::<ChatRequest>(tools).is_err());

    let cache_key = br#"{
            "model":"Qwen3.6-35B-A3B",
            "messages":[{"role":"user","content":"hi"}],
            "prompt_cache_key":"conversation-1"
        }"#;
    assert!(serde_json::from_slice::<ChatRequest>(cache_key).is_ok());
}

#[test]
fn request_token_budget_rejects_overflow() {
    assert_eq!(request_token_budget(10, 20).unwrap(), 31);
    assert!(request_token_budget(1, usize::MAX).is_err());
}

#[test]
fn api_errors_separate_client_and_server_failures() {
    let invalid = ApiError::invalid(anyhow::anyhow!("bad request"));
    assert_eq!(invalid.status(), StatusCode(400));
    assert_eq!(invalid.kind(), "invalid_request_error");
    assert_eq!(invalid.public_message(), "bad request");

    let internal = ApiError::internal(anyhow::anyhow!("secret runtime detail"));
    assert_eq!(internal.status(), StatusCode(500));
    assert_eq!(internal.kind(), "server_error");
    assert_eq!(internal.public_message(), "internal server error");
}
