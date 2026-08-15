use jobs_core::OperationResult;
use runtime_core::{Diagnostic, DiagnosticSeverity};

use crate::contracts::{TextStatisticsRequest, TextStatisticsResult};
use crate::text_stats;

pub fn analyze_text_statistics(
    request: TextStatisticsRequest,
) -> OperationResult<TextStatisticsResult> {
    let stats = text_stats(&request.text);
    let mut diagnostics = Vec::new();
    if request.text.is_empty() {
        diagnostics.push(Diagnostic::new(
            DiagnosticSeverity::Warning,
            "text.empty",
            "Input text is empty.",
        ));
    }

    OperationResult {
        value: Some(TextStatisticsResult {
            byte_count: stats.bytes,
            character_count: stats.chars,
            word_count: stats.words,
            line_count: stats.lines,
            sentence_count: stats.sentences,
        }),
        diagnostics,
        artifacts: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_statistics_operation_returns_shared_result_shape() {
        let result = analyze_text_statistics(TextStatisticsRequest {
            text: "Hello world.\nAgain.".to_string(),
        });
        let value = result.value.expect("operation value");

        assert_eq!(value.word_count, 3);
        assert_eq!(result.diagnostics, Vec::new());
        assert_eq!(result.artifacts, Vec::new());
    }

    #[test]
    fn text_statistics_operation_warns_on_empty_input() {
        let result = analyze_text_statistics(TextStatisticsRequest {
            text: String::new(),
        });

        assert_eq!(result.diagnostics.len(), 1);
        assert_eq!(result.diagnostics[0].code.as_str(), "text.empty");
    }
}
