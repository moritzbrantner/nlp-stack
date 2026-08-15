use text_analysis::{
    analyze_corpus, analyze_text, CorpusAnalysisOptions, DocumentAnalysisOptions, EmbeddingDepth,
    LinguisticDepth,
};
use text_core::TextDocument;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let documents = [
        TextDocument {
            id: "doc-architecture",
            text: "Rust text crates expose stable APIs for lexical analysis, retrieval, and reports.",
            language: Some("en"),
            timestamp: None,
        },
        TextDocument {
            id: "doc-retrieval",
            text: "Retrieval reports combine lexical matches, hashed embeddings, and metadata-aware chunks.",
            language: Some("en"),
            timestamp: None,
        },
        TextDocument {
            id: "doc-retrieval-copy",
            text: "Retrieval reports combine lexical matches, hashed embeddings, and metadata-aware chunks.",
            language: Some("en"),
            timestamp: None,
        },
        TextDocument {
            id: "doc-analysis",
            text: "Corpus analysis highlights keywords, summaries, near duplicates, and semantic neighbors.",
            language: Some("en"),
            timestamp: None,
        },
    ];

    let document_options = DocumentAnalysisOptions {
        keyword_limit: 5,
        summary_sentences: 2,
        linguistic_depth: LinguisticDepth::HeuristicFast,
        embedding_depth: EmbeddingDepth::Hashed {
            dimensions: 64,
            use_idf: false,
        },
        ..DocumentAnalysisOptions::default()
    };
    let document_report = analyze_text(documents[0].id, documents[0].text, &document_options)?;

    println!("Document stats for {}:", document_report.id);
    println!(
        "  words={} sentences={} lexical_density={:.3}",
        document_report.core.stats.basic.words,
        document_report.core.stats.basic.sentences,
        document_report.enriched_stats.lexical_density
    );
    println!("  keywords:");
    for keyword in &document_report.lexical.keywords {
        println!("    {} score={:.3}", keyword.text, keyword.score);
    }
    println!("  extractive summary:");
    for sentence in &document_report.lexical.extractive_summary {
        println!("    {}", sentence.text);
    }
    println!("  diagnostics: {:?}", document_report.diagnostics);

    let corpus_options = CorpusAnalysisOptions {
        document: document_options,
        query: Some("hashed retrieval reports".to_string()),
        top_k: 4,
        tfidf_terms_per_document: 4,
        near_duplicate_threshold: 0.8,
        ..CorpusAnalysisOptions::default()
    };
    let corpus_report = analyze_corpus(documents, &corpus_options)?;

    println!("\nCorpus stats:");
    println!(
        "  documents={} total_terms={} unique_terms={}",
        corpus_report.stats.documents,
        corpus_report.stats.total_terms,
        corpus_report.stats.unique_terms
    );

    println!("  per-document keywords:");
    for document in &corpus_report.documents {
        let keywords = document
            .keywords
            .iter()
            .map(|keyword| keyword.text.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "    {} words={} keywords=[{}]",
            document.id, document.stats.basic.words, keywords
        );
    }

    println!("  near-duplicates:");
    for pair in &corpus_report.near_duplicates {
        println!(
            "    {} <-> {} score={:.3} metric={}",
            pair.left_id, pair.right_id, pair.score, pair.metric
        );
    }

    println!("  semantic neighbors:");
    for pair in &corpus_report.semantic_neighbors {
        println!(
            "    {} <-> {} score={:.3} metric={}",
            pair.left_id, pair.right_id, pair.score, pair.metric
        );
    }

    println!("  diagnostics: {:?}", corpus_report.diagnostics);

    Ok(())
}
