use serde::{Deserialize, Serialize};
use text_core::Result;

use crate::invalid_argument;

use super::{SemanticAnalysisReport, SemanticCluster, SemanticUnit};

/// Backend provenance for optional model-assisted semantic interpretation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticInterpretationMetadata {
    pub backend: String,
    pub model: Option<String>,
}

/// Borrowed deterministic evidence supplied to an interpretation backend.
pub struct SemanticConceptInterpretationRequest<'a> {
    pub cluster: &'a SemanticCluster,
    pub representative: &'a SemanticUnit,
    pub members: Vec<&'a SemanticUnit>,
}

/// Backend-produced content before provenance is attached by `text-analysis`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticConceptInterpretationContent {
    pub label: Option<String>,
    pub summary: Option<String>,
    pub confidence: Option<f32>,
}

/// One optional interpretation annotation over a deterministic concept.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticConceptInterpretation {
    pub cluster_id: String,
    pub representative_unit_id: String,
    pub label: Option<String>,
    pub summary: Option<String>,
    pub confidence: Option<f32>,
    pub metadata: SemanticInterpretationMetadata,
}

/// Optional interpretation annotations attached to an existing semantic report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SemanticInterpretationReport {
    pub concepts: Vec<SemanticConceptInterpretation>,
}

/// Caller-supplied interpretation backend.
///
/// Implementations may use a local model, a hosted model, or deterministic
/// logic, but execution/download/credential/retry policy stays outside
/// `text-analysis`. The deterministic semantic report remains the source of
/// truth regardless of these annotations.
pub trait SemanticInterpretationBackend {
    fn metadata(&self) -> SemanticInterpretationMetadata;

    fn interpret_concept(
        &self,
        request: &SemanticConceptInterpretationRequest<'_>,
    ) -> Result<SemanticConceptInterpretationContent>;
}

/// Applies a caller-supplied optional interpretation backend to deterministic
/// concept clusters without modifying the source semantic structure.
pub fn interpret_semantic_report<B: SemanticInterpretationBackend + ?Sized>(
    report: &SemanticAnalysisReport,
    backend: &B,
) -> Result<SemanticInterpretationReport> {
    let units = report
        .units
        .iter()
        .map(|unit| (unit.id.as_str(), unit))
        .collect::<std::collections::BTreeMap<_, _>>();
    let metadata = backend.metadata();
    if metadata.backend.trim().is_empty() {
        return Err(invalid_argument(
            "semantic interpretation backend name must not be empty",
        ));
    }

    let mut concepts = Vec::with_capacity(report.clusters.len());
    for cluster in &report.clusters {
        let representative = units
            .get(cluster.representative_unit_id.as_str())
            .copied()
            .ok_or_else(|| invalid_argument("semantic cluster representative was not found"))?;
        let members = cluster
            .member_unit_ids
            .iter()
            .filter_map(|id| units.get(id.as_str()).copied())
            .collect::<Vec<_>>();
        if members.len() != cluster.member_unit_ids.len() {
            return Err(invalid_argument(
                "semantic cluster contains an unknown member unit",
            ));
        }
        let content = backend.interpret_concept(&SemanticConceptInterpretationRequest {
            cluster,
            representative,
            members,
        })?;
        if let Some(confidence) = content.confidence {
            if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
                return Err(invalid_argument(
                    "semantic interpretation confidence must be finite and between 0 and 1",
                ));
            }
        }
        concepts.push(SemanticConceptInterpretation {
            cluster_id: cluster.id.clone(),
            representative_unit_id: cluster.representative_unit_id.clone(),
            label: content.label,
            summary: content.summary,
            confidence: content.confidence,
            metadata: metadata.clone(),
        });
    }

    Ok(SemanticInterpretationReport { concepts })
}
