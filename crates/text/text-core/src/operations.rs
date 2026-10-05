use crate::contracts::{TextStatisticsRequest, TextStatisticsResult};
use crate::text_stats;

pub fn analyze_text_statistics(request: TextStatisticsRequest) -> TextStatisticsResult {
    let stats = text_stats(&request.text);
    TextStatisticsResult {
        byte_count: stats.bytes,
        character_count: stats.chars,
        word_count: stats.words,
        line_count: stats.lines,
        sentence_count: stats.sentences,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_statistics_operation_returns_statistics_directly() {
        let result = analyze_text_statistics(TextStatisticsRequest {
            text: "Hello world.\nAgain.".to_string(),
        });

        assert_eq!(result.word_count, 3);
        assert_eq!(result.sentence_count, 2);
    }

    #[test]
    fn text_statistics_operation_handles_empty_input() {
        let result = analyze_text_statistics(TextStatisticsRequest {
            text: String::new(),
        });

        assert_eq!(result.byte_count, 0);
        assert_eq!(result.character_count, 0);
        assert_eq!(result.word_count, 0);
    }
}
