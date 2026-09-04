use text_core::TextSpan;
use unicode_segmentation::UnicodeSegmentation;

const UNICODE_CASES: &[&str] = &[
    "",
    "plain ASCII text",
    "café déjà vu",
    "e\u{301}cole",
    "👍🏽 ok",
    "👨‍👩‍👧‍👦 family",
    "🇩🇪 Deutschland",
    "中文。下一句",
    "مرحبا بالعالم",
    "नमस्ते दुनिया",
    "line one\r\nline two",
];

fn char_boundaries(text: &str) -> Vec<usize> {
    let mut boundaries = text
        .char_indices()
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if boundaries.last().copied() != Some(text.len()) {
        boundaries.push(text.len());
    }
    boundaries
}

#[test]
fn every_utf8_character_boundary_range_constructs_canonically() {
    for text in UNICODE_CASES {
        let boundaries = char_boundaries(text);
        for (start_index, &byte_start) in boundaries.iter().enumerate() {
            for &byte_end in &boundaries[start_index..] {
                let span = TextSpan::from_byte_range(text, byte_start, byte_end)
                    .expect("every ordered pair of UTF-8 character boundaries must be valid");

                assert_eq!((span.byte_start, span.byte_end), (byte_start, byte_end));
                assert_eq!(span.char_start, text[..byte_start].chars().count());
                assert_eq!(span.char_end, text[..byte_end].chars().count());
                assert_eq!(
                    &text[span.byte_start..span.byte_end],
                    &text[byte_start..byte_end]
                );
            }
        }
    }
}

#[test]
fn every_non_character_boundary_is_rejected() {
    for text in UNICODE_CASES {
        for offset in 0..=text.len() {
            if text.is_char_boundary(offset) {
                continue;
            }

            assert!(TextSpan::from_byte_range(text, offset, text.len()).is_err());
            assert!(TextSpan::from_byte_range(text, 0, offset).is_err());
        }
    }
}

#[test]
fn reversed_and_out_of_bounds_ranges_are_rejected() {
    for text in UNICODE_CASES {
        let boundaries = char_boundaries(text);
        if let Some(&last) = boundaries.last() {
            assert!(TextSpan::from_byte_range(text, last.saturating_add(1), last).is_err());
            assert!(TextSpan::from_byte_range(text, 0, last.saturating_add(1)).is_err());
        }

        for &left in &boundaries {
            for &right in &boundaries {
                if left > right {
                    assert!(TextSpan::from_byte_range(text, left, right).is_err());
                }
            }
        }
    }
}

#[test]
fn alternate_coordinates_are_derived_only_from_canonical_bytes() {
    for text in UNICODE_CASES {
        let boundaries = char_boundaries(text);
        for (start_index, &byte_start) in boundaries.iter().enumerate() {
            for &byte_end in &boundaries[start_index..] {
                let canonical = TextSpan::from_byte_range(text, byte_start, byte_end).unwrap();
                let poisoned_legacy = TextSpan {
                    byte_start,
                    byte_end,
                    char_start: usize::MAX,
                    char_end: usize::MAX,
                };

                assert_eq!(canonical.to_utf16(text), poisoned_legacy.to_utf16(text));
                assert_eq!(
                    canonical.to_grapheme(text),
                    poisoned_legacy.to_grapheme(text)
                );

                let utf16 = canonical.to_utf16(text).unwrap();
                assert_eq!(utf16.start, text[..byte_start].encode_utf16().count());
                assert_eq!(utf16.end, text[..byte_end].encode_utf16().count());

                let graphemes = canonical.to_grapheme(text).unwrap();
                assert_eq!(
                    graphemes.start,
                    text[..byte_start].graphemes(true).count()
                );
                assert_eq!(graphemes.end, text[..byte_end].graphemes(true).count());
            }
        }
    }
}
