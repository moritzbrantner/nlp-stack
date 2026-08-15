use std::collections::BTreeMap;

use text_lexical::{CorpusOptions, TextCorpus, TextCorpusDocument};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let documents = [
        corpus_document(
            "architecture",
            "Rust crates expose stable package surfaces for text analysis and retrieval.",
            "en",
            [("source", "guide"), ("kind", "overview")],
        ),
        corpus_document(
            "lexical",
            "TF-IDF and BM25 rank local text documents with deterministic lexical scoring.",
            "en",
            [("source", "guide"), ("kind", "lexical")],
        ),
        corpus_document(
            "retrieval",
            "Hybrid retrieval combines lexical search with hashed semantic embeddings.",
            "en",
            [("source", "guide"), ("kind", "retrieval")],
        ),
        corpus_document(
            "snapshots",
            "Corpus snapshots preserve document identifiers, language tags, and metadata.",
            "en",
            [("source", "guide"), ("kind", "snapshot")],
        ),
    ];

    let corpus = TextCorpus::from_documents(documents, CorpusOptions::default())?;
    let tfidf = corpus.to_tfidf_corpus()?;

    let stats = tfidf.stats();
    println!(
        "Corpus: documents={} total_terms={} unique_terms={} avg_terms={:.2}",
        stats.documents, stats.total_terms, stats.unique_terms, stats.average_terms_per_document
    );

    println!("\nTop corpus terms:");
    for term in tfidf.term_stats(8) {
        println!(
            "  {} count={} documents={} frequency={:.3}",
            term.term, term.collection_count, term.document_count, term.collection_frequency
        );
    }

    println!("\nPer-document TF-IDF terms:");
    for document in &corpus.documents {
        println!("  {}", document.id);
        for term in tfidf.document_tfidf(&document.id, 4)? {
            println!(
                "    {} score={:.3} tf={:.3} idf={:.3}",
                term.term, term.score, term.term_frequency, term.inverse_document_frequency
            );
        }
    }

    println!("\nTF-IDF search: \"hybrid semantic retrieval\"");
    for result in tfidf.search("hybrid semantic retrieval", 3)? {
        println!(
            "  {} score={:.3} matched_terms={}",
            result.id, result.score, result.matched_terms
        );
    }

    let snapshot_json = serde_json::to_string_pretty(&corpus.snapshot()?)?;
    println!("\nSnapshot JSON:\n{snapshot_json}");

    Ok(())
}

fn corpus_document(
    id: &str,
    text: &str,
    language: &str,
    metadata: impl IntoIterator<Item = (&'static str, &'static str)>,
) -> TextCorpusDocument {
    let mut document = TextCorpusDocument::new(id, text);
    document.language = Some(language.to_string());
    document.metadata = metadata
        .into_iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect::<BTreeMap<_, _>>();
    document
}
