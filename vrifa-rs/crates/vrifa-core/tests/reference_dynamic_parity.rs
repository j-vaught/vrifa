mod common;

use common::{load_dynamic_expected, load_dynamic_measurements, load_dynamic_params};
use vrifa_core::reference::{
    compute_dynamic_delta_t_seconds, compute_dynamic_factor, select_dynamic_reference_index,
    DynamicReferenceParams,
};

#[test]
fn dynamic_reference_factor_and_delta_t_match_python_fixture() {
    let measurements = load_dynamic_measurements();
    let params = load_dynamic_params();
    let expected = load_dynamic_expected();
    let factor = compute_dynamic_factor(
        &measurements
            .outer_iter()
            .map(|row| (row[0], row[1]))
            .collect::<Vec<_>>(),
    )
    .expect("dynamic factor is defined");
    let dynamic_params = DynamicReferenceParams {
        factor: Some(factor),
        target_fraction: params.target_fraction,
        lag_scale: params.lag_scale,
        linear_mode: params.linear_mode,
        linear_start: params.linear_start,
        linear_max: params.linear_max,
        total_frames: Some(params.total_frames),
    };
    let delta_t = compute_dynamic_delta_t_seconds(
        params.frame_index,
        params.fps,
        params.roi_pixels,
        &dynamic_params,
    )
    .expect("delta_t is defined");
    let reference_index = select_dynamic_reference_index(
        params.frame_index,
        params.fps,
        params.roi_pixels,
        &dynamic_params,
    );

    let factor_rel = (factor - expected.factor).abs() / expected.factor.abs().max(1.0);
    let delta_t_rel = (delta_t - expected.delta_t).abs() / expected.delta_t.abs().max(1.0);
    assert!(
        factor_rel <= 1e-6,
        "dynamic factor relative diff {factor_rel} exceeded tolerance"
    );
    assert!(
        delta_t_rel <= 1e-6,
        "dynamic delta_t relative diff {delta_t_rel} exceeded tolerance"
    );
    assert_eq!(reference_index, expected.reference_index, "reference index differs");
}
