mod common;

use common::{box_rows, load_contours_rows, load_fixture_config, load_mask, ALL_STAGE_FRAMES};
use vrifa_core::contours::extract_bounding_boxes;

#[test]
fn contours_match_python_goldens_after_sorting() {
    for (input, frame) in ALL_STAGE_FRAMES {
        let config = load_fixture_config(input);
        let mask = load_mask(input, frame);
        let expected = load_contours_rows(input, frame);
        let actual = extract_bounding_boxes(
            &mask,
            config.annotation_segmentation_tolerance,
            config.annotation_segmentation_max_edge_length,
        )
        .expect("contour extraction succeeds");
        assert_eq!(
            box_rows(&actual),
            expected,
            "{input}/frame_{frame:06}/contours differ"
        );
    }
}
