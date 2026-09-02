use super::*;
use crate::chat::Message;

fn conversation() -> Conversation {
    vec![
        Message::new_user("first"),
        Message::new_assistant("older reply"),
        Message::new_user("second"),
        Message::new_assistant("recent reply"),
    ]
}

/// The template wraps assistant turns after the last user message; this
/// index is what tells the nothink rewrite which turns it must cover.
#[test]
fn last_query_index_finds_the_last_real_user_turn() {
    assert_eq!(last_query_index(&conversation()), Some(2));
}

#[test]
fn a_conversation_with_no_user_turn_has_no_query_index() {
    assert_eq!(last_query_index(&vec![Message::new_system("s")]), None);
}
