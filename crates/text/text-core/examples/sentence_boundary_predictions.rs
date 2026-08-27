use std::{env, fs, io};

use serde::{Deserialize, Serialize};
use text_core::{split_sentence_spans_with_abbreviations, TextProcessingOptions};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct BoundaryCase {
    id: String,
    text: String,
    #[serde(default)]
    abbreviations: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct BoundaryPrediction {
    id: String,
    boundary_byte_ends: Vec<usize>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let corpus_path = env::args().nth(1).ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: sentence_boundary_predictions <corpus.jsonl>",
        )
    })?;
    let source = fs::read_to_string(corpus_path)?;
    for raw in source.lines().filter(|line| !line.trim().is_empty()) {
        let case: BoundaryCase = serde_json::from_str(raw)?;
        let abbreviations = case
            .abbreviations
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>();
        let boundary_byte_ends = split_sentence_spans_with_abbreviations(
            &case.text,
            &TextProcessingOptions::default(),
            &abbreviations,
        )
        .into_iter()
        .map(|sentence| sentence.span.byte_end)
        .collect();

        println!(
            "{}",
            serde_json::to_string(&BoundaryPrediction {
                id: case.id,
                boundary_byte_ends,
            })?
        );
    }
    Ok(())
}
