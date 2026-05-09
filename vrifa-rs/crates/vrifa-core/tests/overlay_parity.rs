mod common;

use common::{assert_u8_exact, load_mask, load_overlay, load_source_bgr};
use vrifa_core::overlay::create_overlay;

#[test]
fn overlay_matches_frozen_goldens_for_input_1() {
    for frame in [50, 200, 500] {
        let input = "input_1";
        let frame_bgr = load_source_bgr(input, frame);
        let mask = load_mask(input, frame);
        let expected = load_overlay(input, frame);
        let actual = create_overlay(&frame_bgr, &mask).expect("overlay creation succeeds");
        assert_u8_exact(&format!("{input}/frame_{frame:06}/overlay"), &actual, &expected);
    }
}

#[test]
fn overlay_matches_frozen_goldens_for_input_2() {
    for frame in [30, 60, 90] {
        let input = "input_2";
        let frame_bgr = load_source_bgr(input, frame);
        let mask = load_mask(input, frame);
        let expected = load_overlay(input, frame);
        let actual = create_overlay(&frame_bgr, &mask).expect("overlay creation succeeds");
        assert_u8_exact(&format!("{input}/frame_{frame:06}/overlay"), &actual, &expected);
    }
}
