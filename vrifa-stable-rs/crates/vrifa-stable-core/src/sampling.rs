pub fn choose_annotation_indices(
    total: usize,
    mode: &str,
    count: usize,
    stride: usize,
) -> Vec<usize> {
    if total == 0 {
        return Vec::new();
    }
    match mode {
        "all" => (0..total).collect(),
        "count" => {
            let count = count.min(total);
            if count == 0 {
                Vec::new()
            } else if count == 1 {
                vec![0]
            } else {
                let mut out = Vec::new();
                for i in 0..count {
                    let value =
                        ((i as f64) * ((total - 1) as f64) / ((count - 1) as f64)).floor() as usize;
                    if out.last().copied() != Some(value) {
                        out.push(value);
                    }
                }
                out
            }
        }
        "stride" => {
            if stride == 0 {
                Vec::new()
            } else {
                (0..total).step_by(stride).collect()
            }
        }
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::choose_annotation_indices;

    #[test]
    fn count_mode_matches_numpy_linspace_int_truncation() {
        assert_eq!(
            choose_annotation_indices(10, "count", 4, 1),
            vec![0, 3, 6, 9]
        );
        assert_eq!(choose_annotation_indices(3, "count", 10, 1), vec![0, 1, 2]);
    }
}
