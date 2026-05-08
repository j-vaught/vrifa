mod common;

use common::{
    assert_f32_relative, load_delta, load_fixture_config, load_frame_converted_f32,
    load_peak_before, load_roi_mask,
};
use vrifa_core::delta::compute_delta;

#[test]
fn delta_matches_python_golden_for_input_2_frame_30() {
    let config = load_fixture_config("input_2");
    let roi_mask = load_roi_mask("input_2");
    assert!(config.peak_reference, "input_2 fixtures expect peak-reference mode");

    let frame = 30;
    let frame_converted = load_frame_converted_f32("input_2", frame);
    let peak_before = load_peak_before("input_2", frame);
    let expected = load_delta("input_2", frame);
    let actual = compute_delta(
        &frame_converted,
        &frame_converted,
        &roi_mask,
        &config.channel_weights,
        config.darken_only,
        Some(&peak_before),
    )
    .expect("delta computation succeeds");
    assert_f32_relative("input_2/frame_000030/delta", &actual, &expected, 1e-5);
}
