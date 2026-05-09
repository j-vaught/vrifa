mod common;

use common::{assert_u8_exact, load_delta_norm, load_heatmap, ALL_STAGE_FRAMES};
use vrifa_core::heatmap::apply_turbo_colormap;

#[test]
fn heatmap_matches_frozen_goldens() {
    for (input, frame) in ALL_STAGE_FRAMES {
        let delta_norm = load_delta_norm(input, frame);
        let expected = load_heatmap(input, frame);
        let actual = apply_turbo_colormap(&delta_norm).expect("heatmap creation succeeds");
        assert_u8_exact(&format!("{input}/frame_{frame:06}/heatmap"), &actual, &expected);
    }
}
