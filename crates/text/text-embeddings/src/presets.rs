//! Pinned multilingual embedding model presets (#188).
//!
//! Each preset names a Hugging Face repository, a commit revision and the SHA-256
//! digest of every file the backend reads. Loading resolves the bundle through
//! foundation `model-runtime` and refuses a bundle whose revision or file digests
//! differ from the pins. There is no fallback: a missing or mismatched bundle is an
//! error.

use std::str::FromStr;

use serde::{Deserialize, Serialize};

use crate::{EmbeddingPrompts, PoolingStrategy};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
/// Multilingual (en/de/es and more) sentence-embedding models with pinned weights.
pub enum MultilingualEmbeddingPreset {
    /// `intfloat/multilingual-e5-base`: XLM-RoBERTa base, 768 dimensions, mean
    /// pooling, `query: ` / `passage: ` prefixes. The default (owner decision
    /// 2026-11-03).
    #[serde(rename = "multilingual-e5-base")]
    MultilingualE5Base,
    /// `Qwen/Qwen3-Embedding-0.6B`: Qwen3 decoder, 1024 dimensions, last-token
    /// pooling, instruction-formatted queries. Opt-in.
    #[serde(rename = "qwen3-embedding-0.6b")]
    Qwen3Embedding0_6B,
    /// `BAAI/bge-m3`: XLM-RoBERTa large, 1024 dimensions, CLS pooling, no prefixes
    /// (dense vectors only). Opt-in.
    #[serde(rename = "bge-m3")]
    BgeM3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// One pinned file of a preset bundle.
pub struct EmbeddingPresetFile {
    /// Path inside the Hugging Face repository.
    pub path: &'static str,
    /// Lower-case hex SHA-256 digest of the file content.
    pub sha256: &'static str,
    /// File size in bytes.
    pub size_bytes: u64,
}

const E5_BASE_FILES: &[EmbeddingPresetFile] = &[
    EmbeddingPresetFile {
        path: "config.json",
        sha256: "9dab198f24c8c0879e481cf7822005d5ecbceedbacb390ffafa594e28d31bac4",
        size_bytes: 694,
    },
    EmbeddingPresetFile {
        path: "tokenizer.json",
        sha256: "62c24cdc13d4c9952d63718d6c9fa4c287974249e16b7ade6d5a85e7bbb75626",
        size_bytes: 17_082_660,
    },
    EmbeddingPresetFile {
        path: "model.safetensors",
        sha256: "a18a44fad1d0b46ded15928144138cff1135d5cc8233bdd90be5f18822de09a7",
        size_bytes: 1_112_201_288,
    },
];

const QWEN3_EMBEDDING_0_6B_FILES: &[EmbeddingPresetFile] = &[
    EmbeddingPresetFile {
        path: "config.json",
        sha256: "b5bf1f51fc45be473a54718cef92448d90a1be001bf9b9a44b8c7f10a19feaa9",
        size_bytes: 727,
    },
    EmbeddingPresetFile {
        path: "tokenizer.json",
        sha256: "def76fb086971c7867b829c23a26261e38d9d74e02139253b38aeb9df8b4b50a",
        size_bytes: 11_423_705,
    },
    EmbeddingPresetFile {
        path: "model.safetensors",
        sha256: "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
        size_bytes: 1_191_586_416,
    },
];

const BGE_M3_FILES: &[EmbeddingPresetFile] = &[
    EmbeddingPresetFile {
        path: "config.json",
        sha256: "26159e7ad065073448460117eb24b7a4572f6f4e78eadff65dc0a11c052449fa",
        size_bytes: 687,
    },
    EmbeddingPresetFile {
        path: "tokenizer.json",
        sha256: "21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08",
        size_bytes: 17_098_108,
    },
    EmbeddingPresetFile {
        path: "pytorch_model.bin",
        sha256: "b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38",
        size_bytes: 2_271_145_830,
    },
];

/// The query instruction from Qwen3-Embedding's `config_sentence_transformers.json`.
const QWEN3_QUERY_PROMPT: &str =
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:";
/// The Qwen3-Embedding instruction for symmetric similarity (STS) inputs.
const QWEN3_SYMMETRIC_PROMPT: &str = "Instruct: Retrieve semantically similar text.\nQuery:";

impl MultilingualEmbeddingPreset {
    /// Every preset.
    pub const ALL: &'static [Self] = &[
        Self::MultilingualE5Base,
        Self::Qwen3Embedding0_6B,
        Self::BgeM3,
    ];

    /// The default preset.
    pub const DEFAULT: Self = Self::MultilingualE5Base;

    /// Stable preset id (also the bundle directory name).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MultilingualE5Base => "multilingual-e5-base",
            Self::Qwen3Embedding0_6B => "qwen3-embedding-0.6b",
            Self::BgeM3 => "bge-m3",
        }
    }

    /// Hugging Face repository id; reported as the model id.
    pub fn repo_id(self) -> &'static str {
        match self {
            Self::MultilingualE5Base => "intfloat/multilingual-e5-base",
            Self::Qwen3Embedding0_6B => "Qwen/Qwen3-Embedding-0.6B",
            Self::BgeM3 => "BAAI/bge-m3",
        }
    }

    /// Pinned Hugging Face commit.
    pub fn revision(self) -> &'static str {
        match self {
            Self::MultilingualE5Base => "d128750597153bb5987e10b1c3493a34e5a4502a",
            Self::Qwen3Embedding0_6B => "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
            Self::BgeM3 => "5617a9f61b028005a4858fdac845db406aefb181",
        }
    }

    /// Pinned files with their digests.
    pub fn files(self) -> &'static [EmbeddingPresetFile] {
        match self {
            Self::MultilingualE5Base => E5_BASE_FILES,
            Self::Qwen3Embedding0_6B => QWEN3_EMBEDDING_0_6B_FILES,
            Self::BgeM3 => BGE_M3_FILES,
        }
    }

    /// Pooling the model was trained with.
    pub fn pooling(self) -> PoolingStrategy {
        match self {
            Self::MultilingualE5Base => PoolingStrategy::Mean,
            Self::Qwen3Embedding0_6B => PoolingStrategy::LastToken,
            Self::BgeM3 => PoolingStrategy::Cls,
        }
    }

    /// Per-role prefixes the model expects.
    pub fn prompts(self) -> EmbeddingPrompts {
        match self {
            // e5 model card: "query: " for queries and for symmetric tasks,
            // "passage: " for retrieved passages.
            Self::MultilingualE5Base => EmbeddingPrompts {
                query: "query: ".to_string(),
                passage: "passage: ".to_string(),
                symmetric: "query: ".to_string(),
            },
            Self::Qwen3Embedding0_6B => EmbeddingPrompts {
                query: QWEN3_QUERY_PROMPT.to_string(),
                passage: String::new(),
                symmetric: QWEN3_SYMMETRIC_PROMPT.to_string(),
            },
            Self::BgeM3 => EmbeddingPrompts::none(),
        }
    }

    /// Output vector dimensions.
    pub fn dimensions(self) -> usize {
        match self {
            Self::MultilingualE5Base => 768,
            Self::Qwen3Embedding0_6B | Self::BgeM3 => 1024,
        }
    }

    /// The `model-runtime` bundle spec, pinned to [`Self::revision`].
    #[cfg(feature = "model-bundles")]
    pub fn spec(self) -> model_runtime::HuggingFaceModelSpec {
        self.files().iter().fold(
            model_runtime::HuggingFaceModelSpec::new(
                self.repo_id(),
                model_runtime::ModelTask::TextEmbedding,
            )
            .name(self.as_str())
            .revision(self.revision()),
            |spec, file| spec.file(file.path),
        )
    }
}

impl std::fmt::Display for MultilingualEmbeddingPreset {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for MultilingualEmbeddingPreset {
    type Err = String;

    /// Parses a preset id or Hugging Face repository id.
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let value = value.trim();
        Self::ALL
            .iter()
            .copied()
            .find(|preset| {
                preset.as_str().eq_ignore_ascii_case(value)
                    || preset.repo_id().eq_ignore_ascii_case(value)
            })
            .ok_or_else(|| {
                format!(
                    "unknown embedding preset `{value}`; expected one of {}",
                    Self::ALL
                        .iter()
                        .map(|preset| preset.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })
    }
}

#[cfg(all(feature = "candle", feature = "model-bundles"))]
pub use loading::{load_embedding_preset, verify_bundle_digests, EmbeddingPresetLoadOptions};

#[cfg(all(feature = "candle", feature = "model-bundles"))]
mod loading {
    use std::fs::File;
    use std::io::Read;
    use std::path::PathBuf;

    use model_runtime::{resolve_or_download_bundle, ModelBundle, ModelBundleResolveOptions};
    use sha2::{Digest, Sha256};
    use text_core::{DetectError, Result};

    use super::{EmbeddingPresetFile, MultilingualEmbeddingPreset};
    use crate::{invalid_argument, CandleTextEmbedder};

    #[derive(Debug, Clone, PartialEq, Eq)]
    /// Where preset bundles live and whether they may be downloaded.
    pub struct EmbeddingPresetLoadOptions {
        /// Root directory of `model-runtime` bundles.
        pub bundle_root: PathBuf,
        /// Whether a missing bundle may be downloaded from Hugging Face.
        pub auto_download: bool,
        /// Whether downloads report progress.
        pub download_progress: bool,
        /// Optional Hugging Face token.
        pub hf_token: Option<String>,
    }

    impl Default for EmbeddingPresetLoadOptions {
        /// `TEXT_EMBEDDINGS_MODEL_DIR` (default `.model-runtime`), downloads off.
        fn default() -> Self {
            Self {
                bundle_root: std::env::var_os("TEXT_EMBEDDINGS_MODEL_DIR")
                    .map(PathBuf::from)
                    .unwrap_or_else(|| PathBuf::from(".model-runtime")),
                auto_download: false,
                download_progress: false,
                hf_token: None,
            }
        }
    }

    /// Resolves the pinned bundle, verifies its revision and file digests, and loads
    /// it with the preset's pooling, prompts and L2 normalization.
    pub fn load_embedding_preset(
        preset: MultilingualEmbeddingPreset,
        options: &EmbeddingPresetLoadOptions,
    ) -> Result<CandleTextEmbedder> {
        let resolve = ModelBundleResolveOptions {
            bundle_root: options.bundle_root.clone(),
            auto_download: options.auto_download,
            download_progress: options.download_progress,
            hf_token: options.hf_token.clone(),
            ..ModelBundleResolveOptions::default()
        };
        let bundle = resolve_or_download_bundle(&preset.spec(), &resolve).map_err(|error| {
            DetectError::Source(format!(
                "embedding model `{}` is not available: {error}",
                preset.repo_id()
            ))
        })?;
        if bundle.manifest.revision != preset.revision() {
            return Err(invalid_argument(format!(
                "bundle `{}` has revision `{}`, but the preset pins `{}`",
                preset.as_str(),
                bundle.manifest.revision,
                preset.revision()
            )));
        }
        verify_bundle_digests(&bundle, preset.files())?;
        Ok(CandleTextEmbedder::from_bundle(bundle)?
            .pooling(preset.pooling())
            .prompts(preset.prompts())
            .normalize(true)
            .model_name(preset.repo_id()))
    }

    /// Checks that every pinned file is in the bundle with the pinned size and
    /// SHA-256 digest.
    pub fn verify_bundle_digests(
        bundle: &ModelBundle,
        files: &[EmbeddingPresetFile],
    ) -> Result<()> {
        for file in files {
            let path = bundle.file_path(file.path).ok_or_else(|| {
                invalid_argument(format!(
                    "model bundle `{}` is missing pinned file `{}`",
                    bundle.manifest.name, file.path
                ))
            })?;
            let size = std::fs::metadata(&path)?.len();
            if size != file.size_bytes {
                return Err(invalid_argument(format!(
                    "`{}` in bundle `{}` has {size} bytes, but the pin expects {}",
                    file.path, bundle.manifest.name, file.size_bytes
                )));
            }
            let digest = sha256_file(&path)?;
            if digest != file.sha256 {
                return Err(invalid_argument(format!(
                    "`{}` in bundle `{}` has SHA-256 {digest}, but the pin expects {}",
                    file.path, bundle.manifest.name, file.sha256
                )));
            }
        }
        Ok(())
    }

    fn sha256_file(path: &std::path::Path) -> Result<String> {
        let mut file = File::open(path)?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0_u8; 1 << 20];
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        Ok(hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_parse_by_id_and_repo() {
        for preset in MultilingualEmbeddingPreset::ALL {
            assert_eq!(
                preset.as_str().parse::<MultilingualEmbeddingPreset>(),
                Ok(*preset)
            );
            assert_eq!(
                preset.repo_id().parse::<MultilingualEmbeddingPreset>(),
                Ok(*preset)
            );
        }
        assert!("hashed".parse::<MultilingualEmbeddingPreset>().is_err());
        assert_eq!(
            MultilingualEmbeddingPreset::DEFAULT,
            MultilingualEmbeddingPreset::MultilingualE5Base
        );
    }

    #[test]
    fn every_preset_pins_a_commit_and_file_digests() {
        for preset in MultilingualEmbeddingPreset::ALL {
            let revision = preset.revision();
            assert_eq!(revision.len(), 40, "{preset}");
            assert!(revision.chars().all(|c| c.is_ascii_hexdigit()), "{preset}");
            let paths = preset
                .files()
                .iter()
                .map(|file| file.path)
                .collect::<Vec<_>>();
            assert!(paths.contains(&"config.json"), "{preset}");
            assert!(paths.contains(&"tokenizer.json"), "{preset}");
            for file in preset.files() {
                assert_eq!(file.sha256.len(), 64, "{preset} {}", file.path);
                assert!(
                    file.sha256
                        .chars()
                        .all(|c| matches!(c, '0'..='9' | 'a'..='f')),
                    "{preset} {}",
                    file.path
                );
                assert!(file.size_bytes > 0);
            }
        }
    }

    #[test]
    fn e5_prefixes_queries_and_passages() {
        let prompts = MultilingualEmbeddingPreset::MultilingualE5Base.prompts();
        assert_eq!(
            prompts.apply(crate::EmbeddingInputKind::Query, "Wo liegt Berlin?"),
            "query: Wo liegt Berlin?"
        );
        assert_eq!(
            prompts.apply(crate::EmbeddingInputKind::Passage, "Berlin is in Germany."),
            "passage: Berlin is in Germany."
        );
        let qwen = MultilingualEmbeddingPreset::Qwen3Embedding0_6B.prompts();
        assert!(qwen.query.ends_with("\nQuery:"));
        assert!(qwen.passage.is_empty());
        assert_eq!(
            MultilingualEmbeddingPreset::BgeM3.prompts(),
            EmbeddingPrompts::none()
        );
    }

    #[cfg(feature = "model-bundles")]
    #[test]
    fn specs_request_the_pinned_revision_and_files() {
        for preset in MultilingualEmbeddingPreset::ALL {
            let spec = preset.spec();
            assert_eq!(spec.revision_value(), Some(preset.revision()));
            assert_eq!(spec.repo_id_value(), Some(preset.repo_id()));
            assert_eq!(spec.files.len(), preset.files().len());
        }
    }
}
