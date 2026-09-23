//! No dependencies: also compilable directly with rustc for isolated evidence.
#[path = "../src/surface/fuzzy/distance.rs"]
mod distance;
#[path = "support/fuzzy_distance_legacy.rs"]
mod legacy;

use std::hint::black_box;
use std::time::Instant;

// Keep the paired timing harness explicit; these inputs mirror one distance call.
#[allow(clippy::too_many_arguments)]
fn sample(
    operation: impl Fn(&str, &str, usize) -> Option<usize>,
    left: &str,
    right: &str,
    limit: usize,
    iterations: usize,
) -> f64 {
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(operation(
            black_box(left),
            black_box(right),
            black_box(limit),
        ));
    }
    start.elapsed().as_nanos() as f64 / iterations as f64
}

// Keep fixture construction and both alternating measurement orders auditable together.
#[allow(clippy::too_many_lines)]
fn main() {
    let iterations = std::env::args()
        .skip(1)
        .find(|value| value != "--bench")
        .map(|value| {
            value
                .parse::<usize>()
                .expect("iterations must be an integer")
        })
        .unwrap_or(10_000);
    assert!(iterations > 0, "iterations must be positive");
    for length in [8, 16, 32, 64] {
        let ascii = "a".repeat(length);
        let unicode = "é".repeat(length);
        let fixtures = [
            (
                "one-edit",
                ascii.clone(),
                format!("{}b", "a".repeat(length - 1)),
            ),
            ("rejected", ascii.clone(), "b".repeat(length)),
            ("equal", ascii.clone(), ascii),
            ("unicode", unicode, format!("{}界", "é".repeat(length - 1))),
            (
                "transposition",
                format!("{}ab", "x".repeat(length - 2)),
                format!("{}ba", "x".repeat(length - 2)),
            ),
        ];
        for (fixture, left, right) in fixtures {
            for limit in [1, 2] {
                let expected = legacy::bounded_damerau_levenshtein(&left, &right, limit);
                assert_eq!(
                    distance::bounded_damerau_levenshtein(&left, &right, limit),
                    expected,
                    "fixture={fixture}, length={length}, limit={limit}"
                );
                // Warm both implementations; alternate their order across samples.
                sample(
                    legacy::bounded_damerau_levenshtein,
                    &left,
                    &right,
                    limit,
                    100,
                );
                sample(
                    distance::bounded_damerau_levenshtein,
                    &left,
                    &right,
                    limit,
                    100,
                );
                for sample_index in 0..7 {
                    let (before, after) = if sample_index % 2 == 0 {
                        let before = sample(
                            legacy::bounded_damerau_levenshtein,
                            &left,
                            &right,
                            limit,
                            iterations,
                        );
                        let after = sample(
                            distance::bounded_damerau_levenshtein,
                            &left,
                            &right,
                            limit,
                            iterations,
                        );
                        (before, after)
                    } else {
                        let after = sample(
                            distance::bounded_damerau_levenshtein,
                            &left,
                            &right,
                            limit,
                            iterations,
                        );
                        let before = sample(
                            legacy::bounded_damerau_levenshtein,
                            &left,
                            &right,
                            limit,
                            iterations,
                        );
                        (before, after)
                    };
                    println!(
                        "{{\"fixture\":\"{fixture}\",\"chars\":{length},\"limit\":{limit},\"sample\":{sample_index},\"iterations\":{iterations},\"before_ns_per_call\":{before},\"after_ns_per_call\":{after}}}"
                    );
                }
            }
        }
    }
}
