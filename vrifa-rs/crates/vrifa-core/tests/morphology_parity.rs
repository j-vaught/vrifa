mod common;

use common::{
    assert_f32_max_abs, assert_u8_exact, assert_u8_max_abs, load_binary, load_delta, load_delta_blur,
    load_delta_norm, load_fixture_config, load_mask_pre_lock, load_roi_mask, morphology_params, INPUT_2_FRAMES,
};
use vrifa_core::morphology::detect_mask_from_delta_debug;

#[test]
fn morphology_matches_frozen_goldens_for_input_2() {
    let config = load_fixture_config("input_2");
    let roi_mask = load_roi_mask("input_2");
    let params = morphology_params(&config);

    for frame in INPUT_2_FRAMES {
        let delta = load_delta("input_2", frame);
        let expected_blur = load_delta_blur("input_2", frame);
        let expected_norm = load_delta_norm("input_2", frame);
        let expected_binary = load_binary("input_2", frame);
        let expected_mask = load_mask_pre_lock("input_2", frame);

        let actual =
            detect_mask_from_delta_debug(&delta, &roi_mask, &params).expect("morphology succeeds");
        assert_f32_max_abs(
            &format!("input_2/frame_{frame:06}/delta_blur"),
            &actual.delta_blur,
            &expected_blur,
            1.0,
        );
        assert_u8_max_abs(
            &format!("input_2/frame_{frame:06}/delta_norm"),
            &actual.delta_norm,
            &expected_norm,
            1,
        );
        assert_u8_exact(
            &format!("input_2/frame_{frame:06}/binary"),
            &actual.binary,
            &expected_binary,
        );
        assert_u8_exact(
            &format!("input_2/frame_{frame:06}/mask"),
            &actual.mask,
            &expected_mask,
        );
    }
}
