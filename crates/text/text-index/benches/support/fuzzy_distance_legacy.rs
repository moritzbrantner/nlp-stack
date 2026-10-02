// Frozen function body from aa425fcc8f030731a000f72c736e13d36d2829d3.
// Source blob: 7875625a43c1431aa2e3d29b114051ca4041a3e8.
// Only visibility, lint attributes and signature formatting change. Not production code.

// Keep this frozen baseline lint-independent; only production code should evolve.
#[allow(clippy::needless_range_loop)]
pub(super) fn bounded_damerau_levenshtein(
    left: &str,
    right: &str,
    max_distance: usize,
) -> Option<usize> {
    let left = left.chars().collect::<Vec<_>>();
    let right = right.chars().collect::<Vec<_>>();
    if left.len().abs_diff(right.len()) > max_distance {
        return None;
    }
    if left == right {
        return Some(0);
    }

    let mut previous_previous = vec![0; right.len() + 1];
    let mut previous = (0..=right.len()).collect::<Vec<_>>();
    for (left_index, left_char) in left.iter().enumerate() {
        let row = left_index + 1;
        let mut current = vec![row; right.len() + 1];
        for (right_index, right_char) in right.iter().enumerate() {
            let column = right_index + 1;
            let substitution_cost = usize::from(left_char != right_char);
            let mut distance = (previous[column] + 1)
                .min(current[column - 1] + 1)
                .min(previous[column - 1] + substitution_cost);
            if row > 1
                && column > 1
                && left[row - 1] == right[column - 2]
                && left[row - 2] == right[column - 1]
            {
                distance = distance.min(previous_previous[column - 2] + 1);
            }
            current[column] = distance;
        }
        previous_previous = previous;
        previous = current;
    }
    let distance = previous[right.len()];
    (distance <= max_distance).then_some(distance)
}
