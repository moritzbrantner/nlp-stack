mod derive;
mod dynamics;
mod interpretation;
mod linguistic;
mod model;
mod scaling;
mod units;

use std::collections::BTreeSet;

use text_core::{Result, TextDocument};
use text_embeddings::{HashedTextEmbedder, TextEmbeddingBackend, TextEmbeddingConfig};
use text_lexical::CorpusOptions;

use crate::invalid_argument;

use self::derive::build_report;
use self::units::{conversation_units, document_units};

pub use self::interpretation::{
    interpret_semantic_report, SemanticConceptInterpretation, SemanticConceptInterpretationContent,
    SemanticConceptInterpretationRequest, SemanticInterpretationBackend,
    SemanticInterpretationMetadata, SemanticInterpretationReport,
};
pub use self::linguistic::{
    compose_linguistic_semantic_graph, SemanticGraphEdge, SemanticGraphEdgeKind, SemanticGraphNode,
    SemanticGraphNodeKind, SemanticLinguisticGraph,
};
pub use self::model::{
    ConceptAdoption, ConceptHandoff, ConceptIntroduction, ConversationSemanticDynamics,
    ConversationTurn, RecurringConcept, SemanticAnalysisOptions, SemanticAnalysisReport,
    SemanticCluster, SemanticHotspot, SemanticNeighbor, SemanticTimelinePoint, SemanticUnit,
    SemanticUnitKind, SpeakerConceptShare, SpeakerPairDynamics, SpeakerSemanticProfile,
};
pub use self::scaling::{compare_semantic_neighborhoods, SemanticNeighborhoodEvidence};

/// Analyzes one document with the deterministic hashed embedding baseline.
pub fn analyze_document_semantics(
    document: &TextDocument<'_>,
    options: &SemanticAnalysisOptions,
) -> Result<SemanticAnalysisReport> {
    let embedder = HashedTextEmbedder::new(
        TextEmbeddingConfig {
            dimensions: 128,
            use_idf: false,
        },
        CorpusOptions::default(),
    )?;
    analyze_document_semantics_with(document, options, &embedder)
}

/// Analyzes one document with a caller-supplied embedding backend.
pub fn analyze_document_semantics_with<E: TextEmbeddingBackend + ?Sized>(
    document: &TextDocument<'_>,
    options: &SemanticAnalysisOptions,
    embedder: &E,
) -> Result<SemanticAnalysisReport> {
    validate_options(options)?;
    if document.id.trim().is_empty() {
        return Err(invalid_argument("semantic document id must not be empty"));
    }

    build_report(
        vec![document.id.to_string()],
        SemanticUnitKind::Sentence,
        document_units(document, options),
        options,
        embedder,
    )
}

/// Analyzes an ordered conversation with the deterministic hashed embedding baseline.
pub fn analyze_conversation_semantics(
    turns: &[ConversationTurn<'_>],
    options: &SemanticAnalysisOptions,
) -> Result<SemanticAnalysisReport> {
    let embedder = HashedTextEmbedder::new(
        TextEmbeddingConfig {
            dimensions: 128,
            use_idf: false,
        },
        CorpusOptions::default(),
    )?;
    analyze_conversation_semantics_with(turns, options, &embedder)
}

/// Analyzes an ordered conversation with a caller-supplied embedding backend.
pub fn analyze_conversation_semantics_with<E: TextEmbeddingBackend + ?Sized>(
    turns: &[ConversationTurn<'_>],
    options: &SemanticAnalysisOptions,
    embedder: &E,
) -> Result<SemanticAnalysisReport> {
    validate_options(options)?;
    if turns.is_empty() {
        return Err(invalid_argument(
            "semantic conversation must contain at least one turn",
        ));
    }

    let mut seen = BTreeSet::new();
    for turn in turns {
        if turn.id.trim().is_empty() {
            return Err(invalid_argument("conversation turn id must not be empty"));
        }
        if !seen.insert(turn.id) {
            return Err(invalid_argument(format!(
                "duplicate conversation turn id `{}`",
                turn.id
            )));
        }
    }

    build_report(
        turns.iter().map(|turn| turn.id.to_string()).collect(),
        SemanticUnitKind::SpeakerTurn,
        conversation_units(turns, options),
        options,
        embedder,
    )
}

fn validate_options(options: &SemanticAnalysisOptions) -> Result<()> {
    if options.neighbors_per_unit == 0 {
        return Err(invalid_argument(
            "semantic neighbors_per_unit must be greater than zero",
        ));
    }
    validate_similarity_threshold("neighbor_threshold", options.neighbor_threshold)?;
    validate_similarity_threshold("cluster_threshold", options.cluster_threshold)
}

fn validate_similarity_threshold(name: &str, value: f32) -> Result<()> {
    if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
        return Err(invalid_argument(format!(
            "semantic {name} must be finite and between -1 and 1"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
