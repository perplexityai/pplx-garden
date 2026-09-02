//! Minimal text chat types shared by the tokenizer and OpenAI-compatible API.

use serde::{Deserialize, Serialize};

pub type Conversation = Vec<Message>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Self {
        Self { role, content: content.into() }
    }

    pub fn new_system(content: impl Into<String>) -> Self {
        Self::new(Role::System, content)
    }

    pub fn new_user(content: impl Into<String>) -> Self {
        Self::new(Role::User, content)
    }

    pub fn new_assistant(content: impl Into<String>) -> Self {
        Self::new(Role::Assistant, content)
    }
}
