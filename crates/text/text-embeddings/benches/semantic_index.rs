use criterion::{black_box, criterion_group, criterion_main, Criterion};
use text_embeddings::{HashedTextEmbedder, SemanticTextIndex, TextEmbeddingConfig};
use text_lexical::CorpusOptions;

fn bench_semantic_index(c: &mut Criterion) {
    let mut index = SemanticTextIndex::new(
        HashedTextEmbedder::new(
            TextEmbeddingConfig {
                dimensions: 256,
                use_idf: true,
            },
            CorpusOptions::default(),
        )
        .unwrap(),
    );

    for i in 0..256 {
        let document_id = format!("doc-{i}");
        let document_text = format!("rust cargo pipeline benchmark document {i}");
        index.add_document(document_id, &document_text).unwrap();
    }

    c.bench_function("semantic_text_index_search", |b| {
        b.iter(|| {
            index
                .search(black_box("cargo pipeline status"), 10)
                .unwrap()
        })
    });
}

criterion_group!(benches, bench_semantic_index);
criterion_main!(benches);
