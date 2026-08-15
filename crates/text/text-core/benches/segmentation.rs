use criterion::{black_box, criterion_group, criterion_main, Criterion};
use text_core::{
    build_annotation_graph, detailed_text_stats, split_sentence_spans, tokenize,
    TextProcessingOptions,
};

fn corpus() -> String {
    "Rust analyzes transcripts, captions, scene notes, and search snippets. \
    The tokenizer keeps urls like https://example.test/path, emails like team@example.test, \
    hashtags such as #video, and mentions such as @operator useful for downstream retrieval.\n\n\
    Browser clients call the same segmentation logic through WASM, so this corpus exercises \
    sentence splitting, paragraph splitting, word normalization, punctuation handling, and \
    span construction across repeated production-like text. "
        .repeat(256)
}

fn bench_segmentation(c: &mut Criterion) {
    let text = corpus();
    let options = TextProcessingOptions::default();
    let punctuation_options = TextProcessingOptions {
        include_punctuation: true,
        ..TextProcessingOptions::default()
    };

    c.bench_function("tokenize_120kb", |b| {
        b.iter(|| tokenize(black_box(&text), black_box(&options)))
    });

    c.bench_function("tokenize_with_punctuation_120kb", |b| {
        b.iter(|| tokenize(black_box(&text), black_box(&punctuation_options)))
    });

    c.bench_function("split_sentences_120kb", |b| {
        b.iter(|| split_sentence_spans(black_box(&text), black_box(&options)))
    });

    c.bench_function("detailed_text_stats_120kb", |b| {
        b.iter(|| detailed_text_stats(black_box(&text), black_box(&options)))
    });

    c.bench_function("annotation_graph_120kb", |b| {
        b.iter(|| build_annotation_graph(black_box(&text), black_box(&options)))
    });
}

criterion_group!(benches, bench_segmentation);
criterion_main!(benches);
