//! Three-way exact / cosine LSH / external HNSW evidence on the retrieval
//! workload (#119): identical chunk vectors from `RetrievalIndex`, cosine,
//! top-10 related chunks, single thread.
//!
//! ```text
//! cargo run --release -p moenarch-text-retrieval --features hnsw --example ann_three_way [sizes...]
//! ```
//!
//! Recall and ordering are deterministic; wall-clock numbers describe the
//! machine that ran it and are not a gate.

#[path = "../tests/support/ann_corpus.rs"]
mod ann_corpus;

use std::collections::BTreeSet;
use std::time::Instant;

use vector_analysis_index::approximate::{CosineLshConfig, CosineLshIndex, CosineLshSearchConfig};
use vector_analysis_index::hnsw::{CosineHnswConfig, CosineHnswIndex, CosineHnswSearchConfig};
use vector_analysis_index::{VectorRecord, VectorSearchIndex};

const TOP_K: usize = 10;
const QUERIES: usize = 200;

fn main() {
    let sizes = std::env::args()
        .skip(1)
        .map(|value| value.parse::<usize>().expect("corpus sizes are integers"))
        .collect::<Vec<_>>();
    let sizes = if sizes.is_empty() {
        vec![1_024, 4_096, 16_384]
    } else {
        sizes
    };
    println!(
        "workload: hashed-fixture embedder D={}, {} topics, one chunk per document, seed={:#x}, related chunks top-{TOP_K}, {QUERIES} indexed-chunk queries, single thread",
        ann_corpus::DIMENSIONS,
        ann_corpus::TOPICS,
        ann_corpus::SEED
    );
    for size in sizes {
        let index = ann_corpus::index(size);
        let exact = VectorSearchIndex::import_records(index.vector_records()).unwrap();
        let records = exact.records().to_vec();
        let mut input_hash = 0xcbf2_9ce4_8422_2325_u64;
        for record in &records {
            for value in record.vector.as_slice() {
                for byte in value.to_bits().to_le_bytes() {
                    input_hash = (input_hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
                }
            }
        }
        let queries = records
            .iter()
            .step_by((size / QUERIES).max(1))
            .take(QUERIES)
            .collect::<Vec<_>>();
        println!("\nN={size} vectors_fnv1a64={input_hash:#018x}");
        println!(
            "{:<30} {:>8} {:>6} {:>7} {:>10} {:>10} {:>8}",
            "path", "recall", "top1", "order", "build_ms", "query_us", "scored"
        );

        let started = Instant::now();
        let expected = queries
            .iter()
            .map(|query| {
                exact
                    .search_filtered(query.vector.as_slice(), TOP_K, None)
                    .unwrap()
                    .into_iter()
                    .map(|hit| hit.id.into_string())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let exact_us = started.elapsed().as_secs_f64() * 1e6 / queries.len() as f64;
        report("exact", &expected, &expected, 0.0, exact_us, size as f64);

        for (hash_bits, probe_radius, max_candidates) in [(8, 2, 1_024), (12, 2, 1_024)] {
            let started = Instant::now();
            let lsh = CosineLshIndex::from_records(
                CosineLshConfig {
                    hash_bits,
                    ..CosineLshConfig::default()
                },
                records.iter().cloned(),
            )
            .unwrap();
            let build_ms = started.elapsed().as_secs_f64() * 1e3;
            let config = CosineLshSearchConfig {
                limit: TOP_K,
                probe_radius,
                max_candidates,
            };
            let (actual, query_us, scored) = run(&queries, |query| {
                let report = lsh.search(&query.vector, config).unwrap();
                (
                    report.results.into_iter().map(|r| r.id).collect(),
                    report.candidate_count,
                )
            });
            report(
                &format!("lsh bits={hash_bits} r={probe_radius} C={max_candidates}"),
                &actual,
                &expected,
                build_ms,
                query_us,
                scored,
            );
        }

        let started = Instant::now();
        let hnsw =
            CosineHnswIndex::from_records(CosineHnswConfig::default(), records.iter().cloned())
                .unwrap();
        let build_ms = started.elapsed().as_secs_f64() * 1e3;
        for ef_search in [32, 64] {
            let config = CosineHnswSearchConfig {
                limit: TOP_K,
                ef_search,
            };
            let (actual, query_us, scored) = run(&queries, |query| {
                let report = hnsw.search(&query.vector, config).unwrap();
                (
                    report.results.into_iter().map(|r| r.id).collect(),
                    report.candidate_count,
                )
            });
            report(
                &format!("hnsw M=16 efC=200 ef={ef_search}"),
                &actual,
                &expected,
                build_ms,
                query_us,
                scored,
            );
            if ef_search == 64 && exact_us > query_us {
                println!(
                    "  build amortized after ~{:.0} queries",
                    build_ms * 1e3 / (exact_us - query_us)
                );
            }
        }

        let started = Instant::now();
        let first = index.related_chunks(&queries[0].id, TOP_K).unwrap();
        let first_ms = started.elapsed().as_secs_f64() * 1e3;
        let started = Instant::now();
        for query in &queries {
            index.related_chunks(&query.id, TOP_K).unwrap();
        }
        let warm_us = started.elapsed().as_secs_f64() * 1e6 / queries.len() as f64;
        println!(
            "RetrievalIndex::related_chunks: first call {first_ms:.1} ms ({} hits), warm {warm_us:.1} us/query",
            first.len()
        );
    }
}

fn run(
    queries: &[&VectorRecord],
    mut search: impl FnMut(&VectorRecord) -> (Vec<String>, usize),
) -> (Vec<Vec<String>>, f64, f64) {
    let started = Instant::now();
    let answers = queries
        .iter()
        .map(|query| search(query))
        .collect::<Vec<_>>();
    let query_us = started.elapsed().as_secs_f64() * 1e6 / queries.len() as f64;
    let scored =
        answers.iter().map(|(_, scored)| *scored).sum::<usize>() as f64 / queries.len() as f64;
    (
        answers.into_iter().map(|(ids, _)| ids).collect(),
        query_us,
        scored,
    )
}

fn report(
    path: &str,
    actual: &[Vec<String>],
    expected: &[Vec<String>],
    build_ms: f64,
    query_us: f64,
    scored: f64,
) {
    let mut hits = 0usize;
    let mut top1 = 0usize;
    let mut order = 0usize;
    for (actual, expected) in actual.iter().zip(expected) {
        let expected_ids = expected.iter().collect::<BTreeSet<_>>();
        hits += actual.iter().filter(|id| expected_ids.contains(id)).count();
        top1 += usize::from(actual.first() == expected.first());
        order += usize::from(actual == expected);
    }
    let count = expected.len() as f64;
    println!(
        "{path:<30} {:>8.4} {:>6.3} {:>7.3} {build_ms:>10.1} {query_us:>10.1} {scored:>8.1}",
        hits as f64 / (count * TOP_K as f64),
        top1 as f64 / count,
        order as f64 / count,
    );
}
