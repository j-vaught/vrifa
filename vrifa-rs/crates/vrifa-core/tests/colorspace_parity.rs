mod common;

use common::{assert_u8_exact, colorspace_from_fixture, load_frame_converted_u8, load_source_bgr};
use vrifa_core::colorspace::convert_frame_to_colorspace;

#[test]
fn cielab_conversion_matches_frozen_goldens_for_input_1() {
    for frame in [50, 200, 500] {
        let input = "input_1";
        let source = load_source_bgr(input, frame);
        let expected = load_frame_converted_u8(input, frame);
        let actual = convert_frame_to_colorspace(&source, colorspace_from_fixture(input))
            .expect("colorspace conversion succeeds");
        assert_u8_exact(
            &format!("{input}/frame_{frame:06}/frame_converted"),
            &actual,
            &expected,
        );
    }
}

#[test]
fn cielab_conversion_matches_frozen_goldens_for_input_2() {
    for frame in [30, 60, 90] {
        let input = "input_2";
        let source = load_source_bgr(input, frame);
        let expected = load_frame_converted_u8(input, frame);
        let actual = convert_frame_to_colorspace(&source, colorspace_from_fixture(input))
            .expect("colorspace conversion succeeds");
        assert_u8_exact(
            &format!("{input}/frame_{frame:06}/frame_converted"),
            &actual,
            &expected,
        );
    }
}
