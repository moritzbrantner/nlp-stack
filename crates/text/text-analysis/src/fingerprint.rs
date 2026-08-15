use serde::{Deserialize, Serialize};
use text_core::Result;
use text_core::TextProcessingOptions;
use text_lexical::{character_shingles, token_shingles};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentSimilarityPair {
    pub left_id: String,
    pub right_id: String,
    pub score: f32,
    pub metric: String,
}

pub fn token_shingle_simhash(text: &str, n: usize, options: &TextProcessingOptions) -> Result<u64> {
    let shingles = token_shingles(text, n, options)?;
    Ok(simhash64(
        shingles.iter().map(|shingle| shingle.join("\u{1f}")),
    ))
}

pub fn character_shingle_simhash(text: &str, n: usize) -> Result<u64> {
    let shingles = character_shingles(text, n)?;
    Ok(simhash64(shingles.iter().cloned()))
}

pub fn simhash64<I, S>(items: I) -> u64
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut weights = [0i32; 64];
    for item in items {
        let hash = stable_hash64(item.as_ref().as_bytes());
        for (bit, weight) in weights.iter_mut().enumerate() {
            if (hash >> bit) & 1 == 1 {
                *weight += 1;
            } else {
                *weight -= 1;
            }
        }
    }
    weights.iter().enumerate().fold(0u64, |acc, (bit, weight)| {
        if *weight >= 0 {
            acc | (1u64 << bit)
        } else {
            acc
        }
    })
}

pub fn shingle_hamming_distance(left: u64, right: u64) -> u32 {
    (left ^ right).count_ones()
}

pub fn stable_hash64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}
