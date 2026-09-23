//! Thresholded optimal-string-alignment distance over Unicode scalar values.
//!
//! This preserves the previous routine's adjacent-transposition semantics (OSA,
//! not unrestricted Damerau-Levenshtein). Only cells within `max_distance` of
//! the diagonal can participate in an accepted alignment. Three row buffers
//! are allocated once and rotated; stale cells just outside the band are reset.

pub(super) fn bounded_damerau_levenshtein(
    left: &str,
    right: &str,
    max_distance: usize,
) -> Option<usize> {
    calculate::<false>(left, right, max_distance).0
}

// The const parameter removes accounting from the production instantiation.
// Tests count the actual recurrence evaluations, not a timing approximation.
fn calculate<const COUNT_CELLS: bool>(
    left: &str,
    right: &str,
    max_distance: usize,
) -> (Option<usize>, usize) {
    if left == right {
        return (Some(0), 0);
    }
    let left = left.chars().collect::<Vec<_>>();
    let right = right.chars().collect::<Vec<_>>();
    if left.len().abs_diff(right.len()) > max_distance {
        return (None, 0);
    }
    if left.is_empty() || right.is_empty() {
        return (Some(left.len().max(right.len())), 0);
    }

    let outside = max_distance.saturating_add(1);
    let mut previous_previous = vec![outside; right.len() + 1];
    let mut previous = (0..=right.len())
        .map(|column| column.min(outside))
        .collect::<Vec<_>>();
    let mut current = vec![outside; right.len() + 1];
    let mut cells = 0;

    for row in 1..=left.len() {
        let first = row.saturating_sub(max_distance).max(1);
        let last = row.saturating_add(max_distance).min(right.len());
        current[0] = row.min(outside);
        if first > 1 {
            current[first - 1] = outside;
        }
        if last < right.len() {
            current[last + 1] = outside;
        }
        for column in first..=last {
            if COUNT_CELLS {
                cells += 1;
            }
            let substitution_cost = usize::from(left[row - 1] != right[column - 1]);
            let mut distance = previous[column]
                .saturating_add(1)
                .min(current[column - 1].saturating_add(1))
                .min(previous[column - 1].saturating_add(substitution_cost));
            if row > 1
                && column > 1
                && left[row - 1] == right[column - 2]
                && left[row - 2] == right[column - 1]
            {
                distance = distance.min(previous_previous[column - 2].saturating_add(1));
            }
            current[column] = distance.min(outside);
        }
        std::mem::swap(&mut previous_previous, &mut previous);
        std::mem::swap(&mut previous, &mut current);
    }
    let distance = previous[right.len()];
    ((distance <= max_distance).then_some(distance), cells)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independent full-matrix oracle: no banding, row reuse or threshold pruning.
    fn oracle(left: &str, right: &str) -> usize {
        let left = left.chars().collect::<Vec<_>>();
        let right = right.chars().collect::<Vec<_>>();
        let mut matrix = vec![vec![0; right.len() + 1]; left.len() + 1];
        for (row, values) in matrix.iter_mut().enumerate() {
            values[0] = row;
        }
        for (column, value) in matrix[0].iter_mut().enumerate() {
            *value = column;
        }
        for row in 1..=left.len() {
            for column in 1..=right.len() {
                let cost = usize::from(left[row - 1] != right[column - 1]);
                matrix[row][column] = (matrix[row - 1][column] + 1)
                    .min(matrix[row][column - 1] + 1)
                    .min(matrix[row - 1][column - 1] + cost);
                if row > 1
                    && column > 1
                    && left[row - 1] == right[column - 2]
                    && left[row - 2] == right[column - 1]
                {
                    matrix[row][column] = matrix[row][column].min(matrix[row - 2][column - 2] + 1);
                }
            }
        }
        matrix[left.len()][right.len()]
    }

    fn words(alphabet: &[char], max_length: usize) -> Vec<String> {
        let mut words = vec![String::new()];
        let mut frontier = vec![String::new()];
        for _ in 0..max_length {
            let mut next = Vec::new();
            for prefix in &frontier {
                for character in alphabet {
                    let mut word = prefix.clone();
                    word.push(*character);
                    next.push(word);
                }
            }
            words.extend(next.iter().cloned());
            frontier = next;
        }
        words
    }

    #[test]
    fn exhaustive_unicode_distance_matches_independent_oracle() {
        let words = words(&['a', 'é', '🦀'], 4);
        for left in &words {
            for right in &words {
                let expected = oracle(left, right);
                for limit in 0..=3 {
                    assert_eq!(
                        bounded_damerau_levenshtein(left, right, limit),
                        (expected <= limit).then_some(expected),
                        "left={left:?}, right={right:?}, limit={limit}"
                    );
                }
            }
        }
    }

    #[test]
    fn long_unicode_insertions_deletions_and_transpositions_match_oracle() {
        for length in [8, 16, 32, 64] {
            let original = (0..length)
                .map(|index| ['a', 'é', '🦀'][index % 3])
                .collect::<Vec<_>>();
            let left = original.iter().collect::<String>();
            for position in 0..length {
                let mut variants = Vec::new();
                let mut changed = original.clone();
                changed.remove(position);
                variants.push(changed);
                let mut changed = original.clone();
                changed.insert(position, '界');
                variants.push(changed);
                let mut changed = original.clone();
                changed[position] = '界';
                variants.push(changed);
                if position + 1 < length {
                    let mut changed = original.clone();
                    changed.swap(position, position + 1);
                    variants.push(changed);
                }
                for changed in variants {
                    let right = changed.iter().collect::<String>();
                    let expected = oracle(&left, &right);
                    for limit in 0..=2 {
                        assert_eq!(
                            bounded_damerau_levenshtein(&left, &right, limit),
                            (expected <= limit).then_some(expected),
                            "length={length}, position={position}, limit={limit}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn preserves_osa_semantics_instead_of_silently_changing_the_metric() {
        assert_eq!(bounded_damerau_levenshtein("CA", "ABC", 2), None);
        assert_eq!(bounded_damerau_levenshtein("CA", "ABC", 3), Some(3));
        assert_eq!(bounded_damerau_levenshtein("a", "b", usize::MAX), Some(1));
    }

    #[test]
    fn recurrence_work_is_banded_and_scales_linearly() {
        for length in [8, 16, 32, 64] {
            let left = "a".repeat(length);
            let right = format!("{}b", "a".repeat(length - 1));
            for limit in [1, 2] {
                let (result, cells) = calculate::<true>(&left, &right, limit);
                assert_eq!(result, Some(1));
                assert_eq!(cells, length * (2 * limit + 1) - limit * (limit + 1));
                assert!(cells < length * length);
            }
        }
        assert_eq!(calculate::<true>("identical", "identical", 1), (Some(0), 0));
        assert_eq!(calculate::<true>("a", "length mismatch", 1), (None, 0));
    }
}
