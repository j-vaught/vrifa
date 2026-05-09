mod common;

use ndarray::s;
use common::{assert_u8_exact, load_lock_frames, load_lock_sequence, load_locked_mask};
use vrifa_core::lock::{apply_locking, LockState};

#[test]
fn locking_matches_frozen_fixture_sequence() {
    let sequence = load_lock_sequence();
    let expected = load_locked_mask();
    let lock_frames = load_lock_frames();
    let mut state = LockState::new((sequence.dim().1, sequence.dim().2));
    let mut output = sequence.slice(s![0, .., ..]).to_owned();

    for index in 0..sequence.dim().0 {
        output = apply_locking(
            &sequence.slice(s![index, .., ..]).to_owned(),
            lock_frames,
            Some(&mut state),
        )
        .expect("locking succeeds");
    }

    assert_u8_exact("lock/locked_mask", &output, &expected);
}
