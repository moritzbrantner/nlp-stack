use text_analysis::fingerprint::{
    character_shingle_simhash, shingle_hamming_distance, simhash64, stable_hash64,
    token_shingle_simhash,
};
use text_core::TextProcessingOptions;

#[test]
fn simhash_helpers_are_deterministic_and_distance_based() {
    let left = simhash64(["rust", "text", "analysis"]);
    let same_items = simhash64(["rust", "text", "analysis"]);
    let different = simhash64(["camera", "motion", "scene"]);

    assert_eq!(left, same_items);
    assert_eq!(shingle_hamming_distance(left, left), 0);
    assert_eq!(
        shingle_hamming_distance(left, different),
        shingle_hamming_distance(different, left)
    );
    assert!(shingle_hamming_distance(left, different) > 0);
}

#[test]
fn shingle_simhashes_are_stable_for_token_and_character_inputs() {
    let options = TextProcessingOptions::default();
    let text = "rust crates analyze text with stable shingles";

    let token_hash = token_shingle_simhash(text, 2, &options).unwrap();
    let repeated_token_hash = token_shingle_simhash(text, 2, &options).unwrap();
    let character_hash = character_shingle_simhash(text, 4).unwrap();

    assert_eq!(token_hash, repeated_token_hash);
    assert_ne!(token_hash, character_hash);
    assert!(shingle_hamming_distance(token_hash, character_hash) > 0);
}

#[test]
fn stable_hash64_uses_fnv_style_seed_for_empty_input() {
    assert_eq!(stable_hash64(b""), 0xcbf29ce484222325);
    assert_eq!(stable_hash64(b"rust"), stable_hash64(b"rust"));
    assert_ne!(stable_hash64(b"rust"), stable_hash64(b"text"));
}
