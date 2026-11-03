use text_embeddings::testing::{HashedFixtureEmbedder, HashedFixtureTextIndex};
use text_embeddings::TextEmbeddingConfig;
use text_lexical::CorpusOptions;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let embedder = HashedFixtureEmbedder::new(
        TextEmbeddingConfig {
            dimensions: 128,
            use_idf: true,
        },
        CorpusOptions::default(),
    )?;
    let mut index = HashedFixtureTextIndex::new(embedder);
    index.add_document("scene-1", "opening scene with presenter")?;
    index.add_document("scene-2", "cargo build pipeline status")?;

    for result in index.search("presenter on stage", 2)? {
        println!(
            "{} score={:.3} backend={:?}",
            result.id, result.score, result.metadata.backend
        );
    }

    Ok(())
}
