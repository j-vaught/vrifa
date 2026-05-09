mod common;

use common::{assert_f32_max_abs, load_frame_converted_f32, load_peak_after_3, INPUT_2_FRAMES};
use vrifa_core::peak::update_peak_brightness;

#[test]
fn peak_tracker_matches_frozen_golden_after_three_frames() {
    let expected = load_peak_after_3("input_2");
    let mut peak = None;
    for frame in INPUT_2_FRAMES {
        let frame_converted = load_frame_converted_f32("input_2", frame);
        peak = Some(
            update_peak_brightness(&frame_converted, peak.as_ref())
                .expect("peak update succeeds"),
        );
    }
    let actual = peak.expect("peak map exists after three frames");
    assert_f32_max_abs("input_2/peak_after_3", &actual, &expected, 0.0);
}
