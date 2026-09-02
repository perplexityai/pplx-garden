use super::*;

#[test]
fn q8_is_required_only_for_the_two_checkpoint_exceptions() {
    assert_eq!(
        expected_projection_bits(&["language_model.model.layers.0.mlp.gate"]),
        8
    );
    assert_eq!(
        expected_projection_bits(&[
            "language_model.model.layers.39.mlp.shared_expert_gate"
        ]),
        8
    );
    assert_eq!(
        expected_projection_bits(&[
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj"
        ]),
        4
    );
    assert_eq!(
        expected_projection_bits(&["language_model.model.layers.3.self_attn.q_proj"]),
        4
    );
    assert_eq!(
        expected_projection_bits(&[
            "language_model.model.layers.0.mlp.gate",
            "language_model.model.layers.0.mlp.shared_expert_gate",
        ]),
        4
    );
}
