mod common;

use common::{load_delta_norm, load_fixture_config, load_roi_mask, load_threshold, INPUT_2_FRAMES};
use vrifa_core::threshold::choose_threshold;

#[test]
fn threshold_scalar_matches_python_goldens_for_input_2() {
    let config = load_fixture_config("input_2");
    let roi_mask = load_roi_mask("input_2");

    for frame in INPUT_2_FRAMES {
        let delta_norm = load_delta_norm("input_2", frame);
        let expected = load_threshold("input_2", frame);
        let actual = choose_threshold(
            &delta_norm,
            &roi_mask,
            config.manual_threshold,
            config.percentile_threshold,
            config.threshold_offset,
        )
        .expect("threshold selection succeeds");
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "input_2/frame_{frame:06}/threshold differs"
        );
    }
}
