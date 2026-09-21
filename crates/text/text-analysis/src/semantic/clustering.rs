use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Globally strongest average-link merges, with ties resolved by source order.
/// Cached weighted averages and lazy invalidation bound work to O(n² log n)
/// and storage to O(n²), including candidates superseded by earlier merges.
pub(super) fn average_link_clusters(similarities: &[Vec<f32>], threshold: f32) -> Vec<Vec<usize>> {
    let mut members = (0..similarities.len()).map(|i| vec![i]).collect::<Vec<_>>();
    let mut scores = similarities
        .iter()
        .map(|row| {
            row.iter()
                .map(|&score| f64::from(score))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let threshold = f64::from(threshold);
    let mut candidates = Vec::new();
    for (left, row) in scores.iter().enumerate() {
        for (right, &score) in row.iter().enumerate().skip(left + 1) {
            if score >= threshold {
                candidates.push(ClusterPair {
                    score,
                    left,
                    right,
                    left_size: 1,
                    right_size: 1,
                });
            }
        }
    }
    let mut candidates = BinaryHeap::from(candidates);
    let mut remaining_clusters = members.len();

    while let Some(pair) = candidates.pop() {
        let ClusterPair {
            left,
            right,
            left_size,
            right_size,
            ..
        } = pair;
        // A cluster grows strictly on merging; its size also identifies its revision.
        if members[left].len() != left_size || members[right].len() != right_size {
            continue;
        }
        let merged_size = left_size + right_size;
        let right_members = std::mem::take(&mut members[right]);
        members[left].extend(right_members);
        remaining_clusters -= 1;
        if remaining_clusters == 1 {
            break;
        }
        for (other, other_members) in members.iter().enumerate() {
            if other == left || other_members.is_empty() {
                continue;
            }
            let score = (scores[left][other] * left_size as f64
                + scores[right][other] * right_size as f64)
                / merged_size as f64;
            scores[left][other] = score;
            scores[other][left] = score;
            if score >= threshold {
                let (first, second) = (left.min(other), left.max(other));
                candidates.push(ClusterPair {
                    score,
                    left: first,
                    right: second,
                    left_size: members[first].len(),
                    right_size: members[second].len(),
                });
            }
        }
    }

    // Keeping the left slot preserves the original source-order tie breaker.
    members.retain(|cluster| !cluster.is_empty());
    for cluster in &mut members {
        cluster.sort_unstable();
    }
    members
}

#[derive(Debug)]
struct ClusterPair {
    score: f64,
    left: usize,
    right: usize,
    left_size: usize,
    right_size: usize,
}

impl Ord for ClusterPair {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| other.left.cmp(&self.left))
            .then_with(|| other.right.cmp(&self.right))
            .then_with(|| self.left_size.cmp(&other.left_size))
            .then_with(|| self.right_size.cmp(&other.right_size))
    }
}

impl PartialOrd for ClusterPair {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ClusterPair {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for ClusterPair {}

#[cfg(test)]
mod tests {
    use super::average_link_clusters;

    // Deliberately slow definition of average linkage, independent of the cache/heap.
    fn reference(similarities: &[Vec<f32>], threshold: f32) -> Vec<Vec<usize>> {
        let mut clusters = (0..similarities.len()).map(|i| vec![i]).collect::<Vec<_>>();
        loop {
            let mut best = None;
            for left in 0..clusters.len() {
                for right in left + 1..clusters.len() {
                    let sum: f64 = clusters[left]
                        .iter()
                        .flat_map(|&a| {
                            clusters[right]
                                .iter()
                                .map(move |&b| f64::from(similarities[a][b]))
                        })
                        .sum();
                    let score = sum / (clusters[left].len() * clusters[right].len()) as f64;
                    if score >= f64::from(threshold)
                        && best.is_none_or(|(_, _, previous)| score > previous)
                    {
                        best = Some((left, right, score));
                    }
                }
            }
            let Some((left, right, _)) = best else {
                break;
            };
            let removed = clusters.remove(right);
            clusters[left].extend(removed);
        }
        for cluster in &mut clusters {
            cluster.sort_unstable();
        }
        clusters
    }

    #[test]
    fn cached_linkage_matches_definition_including_ties_and_stale_candidates() {
        for size in 0..20 {
            for seed in 0..12 {
                // Dyadic similarities exercise ties, uneven cluster sizes, and disconnected groups.
                let matrix = (0..size)
                    .map(|a| {
                        (0..size)
                            .map(|b| {
                                if a == b {
                                    1.0
                                } else {
                                    ((a.min(b) * 17 + a.max(b) * 31 + seed * 7) % 17) as f32 / 8.0
                                        - 1.0
                                }
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                for threshold in [-1.0, 0.0, 0.25, 0.5, 0.75, 1.0] {
                    assert_eq!(
                        average_link_clusters(&matrix, threshold),
                        reference(&matrix, threshold),
                        "size={size}, seed={seed}, threshold={threshold}"
                    );
                }
            }
        }
    }
}
