//! A deliberately small, serial OpenAI-compatible chat-completions server.

mod session;

use std::io::Read as _;
use std::net::ToSocketAddrs;
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context as _, Result, ensure};
use serde::{Deserialize, Serialize};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

use crate::chat::{Conversation, Message, Role};
use crate::generate::{Generator, Thinking};
use crate::metal::MetalContext;
use crate::model::{Qwen3_5Model, Scratch};
use session::SessionStore;

pub const MODEL_ID: &str = "Qwen3.6-35B-A3B";
const MAX_REQUEST_BYTES: usize = 1 << 20;
const SESSION_CACHE_ENTRIES: usize = 2;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ChatRequest {
    model: String,
    messages: Conversation,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    prompt_cache_key: Option<String>,
}

fn default_max_tokens() -> usize {
    128
}

fn request_token_budget(prompt_tokens: usize, max_tokens: usize) -> Result<usize> {
    prompt_tokens
        .checked_add(max_tokens)
        .and_then(|tokens| tokens.checked_add(1))
        .context("request token count overflow")
}

#[derive(Serialize)]
struct ChatResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: &'static str,
    choices: [Choice; 1],
    usage: Usage,
}

#[derive(Serialize)]
struct Choice {
    index: usize,
    message: Message,
    finish_reason: &'static str,
}

#[derive(Serialize, Clone, Copy)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
    prompt_tokens_details: PromptTokensDetails,
}

#[derive(Serialize, Clone, Copy)]
struct PromptTokensDetails {
    cached_tokens: usize,
}

#[derive(Serialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Serialize)]
struct ErrorBody {
    message: String,
    #[serde(rename = "type")]
    kind: &'static str,
}

enum ApiError {
    InvalidRequest(anyhow::Error),
    Internal(anyhow::Error),
}

impl ApiError {
    fn invalid(error: anyhow::Error) -> Self {
        Self::InvalidRequest(error)
    }

    fn internal(error: anyhow::Error) -> Self {
        Self::Internal(error)
    }

    fn status(&self) -> StatusCode {
        match self {
            Self::InvalidRequest(_) => StatusCode(400),
            Self::Internal(_) => StatusCode(500),
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Self::InvalidRequest(_) => "invalid_request_error",
            Self::Internal(_) => "server_error",
        }
    }

    fn public_message(&self) -> String {
        match self {
            Self::InvalidRequest(error) => format!("{error:#}"),
            Self::Internal(_) => "internal server error".to_string(),
        }
    }

    fn source(&self) -> &anyhow::Error {
        match self {
            Self::InvalidRequest(error) | Self::Internal(error) => error,
        }
    }
}

type ApiResult<T> = std::result::Result<T, ApiError>;
type JsonResponse = Response<std::io::Cursor<Vec<u8>>>;

struct Engine {
    ctx: MetalContext,
    model: Qwen3_5Model,
    generator: Generator,
    sessions: SessionStore,
    scratch: Scratch,
    max_seq: usize,
    next_id: u64,
}

impl Engine {
    fn load(model_dir: &Path, max_seq: usize) -> Result<Self> {
        let ctx = MetalContext::new()?;
        let model = Qwen3_5Model::load(&ctx, model_dir)?;
        let declared = model.config.max_position_embeddings;
        let max_seq = if declared == 0 { max_seq } else { max_seq.min(declared) };
        ensure!(max_seq > 1, "max_seq must be at least 2");
        let mut generator = Generator::from_model_dir(model_dir)?;
        generator.add_stop_tokens(&model.config.eos_token_id.as_vec());
        let scratch = model.new_scratch_with_capacity(&ctx, max_seq)?;
        Ok(Self {
            ctx,
            model,
            generator,
            sessions: SessionStore::new(max_seq, SESSION_CACHE_ENTRIES),
            scratch,
            max_seq,
            next_id: 1,
        })
    }

    fn validate_request(&self, request: &ChatRequest) -> Result<Vec<u32>> {
        ensure!(
            request.model == MODEL_ID,
            "unknown model {:?}; this server exposes only {MODEL_ID}",
            request.model
        );
        ensure!(!request.stream, "streaming is not supported");
        ensure!(!request.messages.is_empty(), "messages must not be empty");
        ensure!(
            request.messages.last().is_some_and(|m| m.role == Role::User),
            "the final message must have role=user"
        );
        ensure!(request.max_tokens > 0, "max_tokens must be greater than zero");

        // The minimal API serves direct answers. The tokenizer's disabled-
        // thinking history rewrite also keeps multi-turn prompts prefix-cacheable.
        let prompt_ids =
            self.generator.encode_chat(&request.messages, Thinking::Disabled)?;
        ensure!(!prompt_ids.is_empty(), "chat template produced an empty prompt");
        let request_tokens =
            request_token_budget(prompt_ids.len(), request.max_tokens)?;
        ensure!(
            request_tokens <= self.max_seq,
            "request needs {} tokens but server max_seq is {}",
            request_tokens,
            self.max_seq
        );
        Ok(prompt_ids)
    }

    fn complete(&mut self, request: ChatRequest) -> ApiResult<ChatResponse> {
        let prompt_ids = self.validate_request(&request).map_err(ApiError::invalid)?;
        self.complete_validated(request, prompt_ids).map_err(ApiError::internal)
    }

    fn complete_validated(
        &mut self,
        request: ChatRequest,
        prompt_ids: Vec<u32>,
    ) -> Result<ChatResponse> {
        let (mut session, cached_tokens) = self.sessions.acquire(
            &self.ctx,
            &self.model,
            &prompt_ids,
            request.prompt_cache_key.as_deref(),
        )?;
        let suffix = &prompt_ids[cached_tokens..];
        ensure!(!suffix.is_empty(), "session cache returned an exact prompt match");
        let generation = self.generator.generate(
            &self.ctx,
            &self.model,
            &mut session.state,
            &mut self.scratch,
            suffix,
            request.max_tokens,
        )?;
        let decode_fed = generation
            .fed
            .checked_sub(suffix.len())
            .context("decode state advanced less than the prompt suffix")?;
        ensure!(
            decode_fed <= generation.tokens.len(),
            "decode state advanced beyond returned greedy tokens"
        );
        session.tokens.extend_from_slice(suffix);
        session.tokens.extend_from_slice(&generation.tokens[..decode_fed]);
        ensure!(
            session.state.pos == session.tokens.len(),
            "session token/state position mismatch"
        );
        self.sessions.release(session, request.prompt_cache_key.as_deref());
        let finish_reason = if generation.stopped { "stop" } else { "length" };
        let completion_tokens = generation.tokens.len();
        let created = now();
        let id = format!("chatcmpl-{}-{}", created, self.next_id);
        self.next_id += 1;
        Ok(ChatResponse {
            id,
            object: "chat.completion",
            created,
            model: MODEL_ID,
            choices: [Choice {
                index: 0,
                message: Message::new_assistant(generation.text),
                finish_reason,
            }],
            usage: Usage {
                prompt_tokens: prompt_ids.len(),
                completion_tokens,
                total_tokens: prompt_ids.len() + completion_tokens,
                prompt_tokens_details: PromptTokensDetails { cached_tokens },
            },
        })
    }
}

pub fn run(model_dir: &Path, bind: &str, max_seq: usize) -> Result<()> {
    let address = bind
        .to_socket_addrs()
        .with_context(|| format!("resolving bind address {bind}"))?
        .next()
        .with_context(|| format!("bind address {bind} resolved to nothing"))?;
    let server = Server::http(address)
        .map_err(|e| anyhow::anyhow!("{e}"))
        .with_context(|| format!("binding http://{bind}"))?;
    let mut engine = Engine::load(model_dir, max_seq)?;
    eprintln!("serving {MODEL_ID} on http://{address}");

    for mut request in server.incoming_requests() {
        let started = Instant::now();
        let method = request.method().as_str().to_string();
        let path = request.url().to_string();
        let result = dispatch(&mut engine, &mut request);
        let (response, usage, error) = match result {
            Ok((response, usage)) => (response, usage, None),
            Err(error) => {
                if matches!(&error, ApiError::Internal(_)) {
                    eprintln!("request failed: {:#}", error.source());
                }
                let message = error.public_message();
                let response = json_response(
                    error.status(),
                    &ErrorEnvelope {
                        error: ErrorBody {
                            message: message.clone(),
                            kind: error.kind(),
                        },
                    },
                )?;
                (response, None, Some(message))
            }
        };
        eprintln!(
            "{}",
            request_log_line(
                &method,
                &path,
                response.status_code().0,
                started.elapsed(),
                usage.as_ref(),
                error.as_deref(),
            )
        );
        if let Err(error) = request.respond(response) {
            eprintln!("response error: {error}");
        }
    }
    Ok(())
}

/// One stderr line per request: method, path, status, wall time, and for chat
/// completions the token counts plus request-level tokens per second (wall
/// time includes prefill and any queue wait, so this is not a decode rate).
fn request_log_line(
    method: &str,
    path: &str,
    status: u16,
    elapsed: Duration,
    usage: Option<&Usage>,
    error: Option<&str>,
) -> String {
    let mut line = format!("{method} {path} {status} {}ms", elapsed.as_millis());
    if let Some(usage) = usage {
        let secs = elapsed.as_secs_f64().max(1e-3);
        line.push_str(&format!(
            " prompt={} cached={} completion={} tok/s={:.1}",
            usage.prompt_tokens,
            usage.prompt_tokens_details.cached_tokens,
            usage.completion_tokens,
            usage.completion_tokens as f64 / secs
        ));
    }
    if let Some(error) = error {
        line.push_str(&format!(" error={error:?}"));
    }
    line
}

/// Routes one request; chat completions also return their usage for the log.
fn dispatch(
    engine: &mut Engine,
    request: &mut Request,
) -> ApiResult<(JsonResponse, Option<Usage>)> {
    let path = request.url().split('?').next().unwrap_or(request.url());
    match (request.method(), path) {
        (&Method::Get, "/health") => {
            json_response(StatusCode(200), &serde_json::json!({"status": "ok"}))
                .map(|response| (response, None))
                .map_err(ApiError::internal)
        }
        (&Method::Get, "/v1/models") => json_response(
            StatusCode(200),
            &serde_json::json!({
                "object": "list",
                "data": [{
                    "id": MODEL_ID,
                    "object": "model",
                    "created": 0,
                    "owned_by": "lily"
                }]
            }),
        )
        .map(|response| (response, None))
        .map_err(ApiError::internal),
        (&Method::Post, "/v1/chat/completions") => {
            let mut body = Vec::new();
            request
                .as_reader()
                .take((MAX_REQUEST_BYTES + 1) as u64)
                .read_to_end(&mut body)
                .context("reading request body")
                .map_err(ApiError::internal)?;
            if body.len() > MAX_REQUEST_BYTES {
                return Err(ApiError::invalid(anyhow::anyhow!(
                    "request body exceeds {MAX_REQUEST_BYTES} bytes"
                )));
            }
            let chat: ChatRequest = serde_json::from_slice(&body)
                .context("parsing chat request")
                .map_err(ApiError::invalid)?;
            let response = engine.complete(chat)?;
            let http = json_response(StatusCode(200), &response)
                .map_err(ApiError::internal)?;
            Ok((http, Some(response.usage)))
        }
        _ => json_response(
            StatusCode(404),
            &ErrorEnvelope {
                error: ErrorBody {
                    message: "not found".to_string(),
                    kind: "invalid_request_error",
                },
            },
        )
        .map(|response| (response, None))
        .map_err(ApiError::internal),
    }
}

fn json_response<T: Serialize>(status: StatusCode, value: &T) -> Result<JsonResponse> {
    let body = serde_json::to_vec(value)?;
    let content_type =
        Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..])
            .map_err(|_| anyhow::anyhow!("invalid static content-type header"))?;
    Ok(Response::from_data(body).with_status_code(status).with_header(content_type))
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

#[cfg(test)]
#[path = "../tests/unit/serve.rs"]
mod tests;
