//! Stanza oracle adapter (feature `stanza`): the local `sidecars/stanza` HTTP/JSON
//! sidecar as a [`SyntaxBackend`].
//!
//! The sidecar runs pinned Stanza pipelines (version and per-language model
//! packages pinned by digest in `sidecars/stanza/models.lock.json`). Stanza
//! tokenizes each sentence, including multi-word token expansion, and reports
//! code-point offsets relative to the sentence; this adapter converts them to exact
//! UTF-8 half-open byte spans in the source text and refuses a response whose
//! offsets do not reproduce the token text.
//!
//! [`StanzaSidecar`] talks to a running sidecar. [`RecordedStanzaResponse`] decodes
//! a response obtained elsewhere (a committed recording in tests, or a browser that
//! called the sidecar itself) through the same path. Neither falls back to anything:
//! an unreachable sidecar, a missing model or an unserved language is an error.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use text_core::TextSpan;

use crate::morphology::{MorphTagSet, PosTag};
use crate::syntax::DependencyRelation;
use crate::syntax_backend::{
    SyntaxBackend, SyntaxBackendError, SyntaxModelVersion, SyntaxProvenance, SyntaxRequest,
    SyntaxSentence, SyntaxToken, SyntaxWord,
};
use crate::ud_model::parse_ud_feats;

/// The sidecar protocol version this adapter speaks.
pub const STANZA_PROTOCOL_VERSION: u32 = 1;
/// The Stanza release the sidecar pins (`sidecars/stanza/pyproject.toml`).
pub const STANZA_VERSION: &str = "1.15.0";
/// Languages the sidecar's models lock covers.
pub const STANZA_LANGUAGES: &[&str] = &["en", "de", "es"];
/// The sidecar's default address.
pub const STANZA_DEFAULT_ENDPOINT: &str = "http://127.0.0.1:8765";
const BACKEND: &str = "stanza";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
/// One sentence of a `POST /v1/analyze` body.
pub struct StanzaSentenceRequest {
    /// The caller's sentence index.
    pub id: usize,
    /// ISO 639-1 language.
    pub language: String,
    /// The sentence text.
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
/// A `POST /v1/analyze` body.
pub struct StanzaAnalyzeRequest {
    /// The sentences, each with its routed language.
    pub sentences: Vec<StanzaSentenceRequest>,
}

impl StanzaAnalyzeRequest {
    /// The body for `request`.
    pub fn new(request: &SyntaxRequest<'_>) -> Self {
        Self {
            sentences: request
                .sentences
                .iter()
                .map(|sentence| StanzaSentenceRequest {
                    id: sentence.sentence_index,
                    language: sentence.language.clone(),
                    text: request.sentence_text(sentence).to_string(),
                })
                .collect(),
        }
    }
}

#[derive(Debug, Deserialize)]
struct WireResponse {
    protocol: u32,
    stanza_version: String,
    sentences: Vec<WireSentence>,
}

#[derive(Debug, Deserialize)]
struct WireSentence {
    id: usize,
    language: String,
    models: WireModels,
    tokens: Vec<WireToken>,
}

#[derive(Debug, Deserialize)]
struct WireModels {
    repository: String,
    revision: String,
    packages: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct WireToken {
    start_char: usize,
    end_char: usize,
    text: String,
    words: Vec<WireWord>,
}

#[derive(Debug, Deserialize)]
struct WireWord {
    id: usize,
    text: String,
    lemma: Option<String>,
    upos: Option<String>,
    xpos: Option<String>,
    feats: Option<String>,
    head: usize,
    deprel: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WireErrorBody {
    error: WireError,
}

#[derive(Debug, Deserialize)]
struct WireError {
    kind: String,
    message: String,
    #[serde(default)]
    language: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
/// `GET /v1/health`: what the sidecar serves.
pub struct StanzaHealth {
    /// Protocol version.
    pub protocol: u32,
    /// Installed Stanza version.
    pub stanza_version: String,
    /// Per-language readiness and pinned models.
    pub languages: BTreeMap<String, StanzaLanguageHealth>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
/// One language's readiness.
pub struct StanzaLanguageHealth {
    /// Whether every pinned model file is present with its digest.
    pub ready: bool,
    /// Why not, when not ready.
    #[serde(default)]
    pub problems: Vec<String>,
    /// Model repository (`stanfordnlp/stanza-de`).
    pub repository: Option<String>,
    /// Pinned model revision.
    pub revision: Option<String>,
    /// Stage to package.
    #[serde(default)]
    pub packages: BTreeMap<String, String>,
}

fn protocol(message: impl Into<String>) -> SyntaxBackendError {
    SyntaxBackendError::Protocol {
        backend: BACKEND.to_string(),
        message: message.into(),
    }
}

/// Maps a sidecar error body (any non-200 status) to the typed error.
fn decode_error(status: u16, body: &str) -> SyntaxBackendError {
    let backend = BACKEND.to_string();
    match serde_json::from_str::<WireErrorBody>(body) {
        Ok(WireErrorBody { error }) => match error.kind.as_str() {
            "model_missing" => SyntaxBackendError::ModelMissing {
                backend,
                language: error.language.unwrap_or_default(),
                message: error.message,
            },
            "unsupported_language" => SyntaxBackendError::UnsupportedLanguage {
                backend,
                language: error.language.unwrap_or_default(),
            },
            "internal" => SyntaxBackendError::Failed {
                backend,
                message: error.message,
            },
            _ => protocol(format!("HTTP {status} {}: {}", error.kind, error.message)),
        },
        Err(_) => protocol(format!("HTTP {status} without an error body")),
    }
}

/// Decodes a `POST /v1/analyze` response for `request`. Each response sentence must
/// answer the request sentence at the same position, in the requested language, and
/// every token's offsets must reproduce its text in the source.
pub fn decode_stanza_response(
    request: &SyntaxRequest<'_>,
    body: &str,
) -> Result<Vec<SyntaxSentence>, SyntaxBackendError> {
    let response: WireResponse = serde_json::from_str(body)
        .map_err(|error| protocol(format!("malformed analyze response: {error}")))?;
    if response.protocol != STANZA_PROTOCOL_VERSION {
        return Err(protocol(format!(
            "sidecar speaks protocol {}, expected {STANZA_PROTOCOL_VERSION}",
            response.protocol
        )));
    }
    if response.stanza_version != STANZA_VERSION {
        return Err(protocol(format!(
            "sidecar runs Stanza {}, this adapter pins {STANZA_VERSION}; rerun sidecars/stanza/setup.sh",
            response.stanza_version
        )));
    }
    if response.sentences.len() != request.sentences.len() {
        return Err(protocol(format!(
            "{} sentences requested, {} answered",
            request.sentences.len(),
            response.sentences.len()
        )));
    }
    request
        .sentences
        .iter()
        .zip(response.sentences)
        .map(|(wanted, sentence)| {
            if sentence.id != wanted.sentence_index || sentence.language != wanted.language {
                return Err(protocol(format!(
                    "answer for sentence {} ({}) does not match request sentence {} ({})",
                    sentence.id, sentence.language, wanted.sentence_index, wanted.language
                )));
            }
            decode_sentence(
                request.text,
                wanted.span,
                &response.stanza_version,
                sentence,
            )
        })
        .collect()
}

fn decode_sentence(
    text: &str,
    span: TextSpan,
    stanza_version: &str,
    sentence: WireSentence,
) -> Result<SyntaxSentence, SyntaxBackendError> {
    let sentence_text = &text[span.byte_start..span.byte_end];
    // Byte offset of every code point boundary (Python string indices).
    let mut boundaries = sentence_text
        .char_indices()
        .map(|(offset, _)| offset)
        .collect::<Vec<_>>();
    boundaries.push(sentence_text.len());
    let byte = |char_offset: usize| {
        boundaries
            .get(char_offset)
            .map(|offset| span.byte_start + offset)
            .ok_or_else(|| {
                protocol(format!(
                    "code-point offset {char_offset} is past the sentence"
                ))
            })
    };

    let mut tokens = Vec::with_capacity(sentence.tokens.len());
    let mut words = Vec::new();
    // Stanza word id (1-based) -> index in `words`.
    let mut word_index = BTreeMap::new();
    let mut heads = Vec::new();
    for (token_index, token) in sentence.tokens.into_iter().enumerate() {
        let token_span =
            TextSpan::from_byte_range(text, byte(token.start_char)?, byte(token.end_char)?)
                .map_err(|error| protocol(format!("token {token_index}: {error}")))?;
        let first_word = words.len();
        for word in token.words {
            word_index.insert(word.id, words.len());
            heads.push(word.head);
            let upos = word.upos.as_deref().and_then(PosTag::from_ud);
            let (relation, subtype) = word
                .deprel
                .as_deref()
                .map(DependencyRelation::parse_ud_deprel)
                .unwrap_or((DependencyRelation::Dep, None));
            words.push(SyntaxWord {
                token: token_index,
                form: word.text,
                lemma: word.lemma,
                upos,
                xpos: word.xpos,
                feats: word
                    .feats
                    .as_deref()
                    .map(parse_ud_feats)
                    .unwrap_or_else(MorphTagSet::new),
                head: None,
                relation,
                subtype,
            });
        }
        tokens.push(SyntaxToken {
            span: token_span,
            text: token.text,
            first_word,
            word_count: words.len() - first_word,
        });
    }
    for (word, head) in words.iter_mut().zip(heads) {
        word.head = match head {
            0 => None,
            id => Some(
                *word_index
                    .get(&id)
                    .ok_or_else(|| protocol(format!("head {id} is not a word of the sentence")))?,
            ),
        };
    }
    let models = sentence
        .models
        .packages
        .into_iter()
        .map(|(stage, package)| SyntaxModelVersion {
            stage,
            package,
            source: sentence.models.repository.clone(),
            revision: sentence.models.revision.clone(),
        })
        .collect();
    let result = SyntaxSentence {
        sentence_index: sentence.id,
        language: sentence.language,
        provenance: SyntaxProvenance {
            backend: BACKEND.to_string(),
            backend_version: stanza_version.to_string(),
            models,
        },
        tokens,
        words,
    };
    result.validate(text, span).map_err(protocol)?;
    Ok(result)
}

/// A Stanza response obtained outside this process (a committed recording, or a
/// browser that called the sidecar itself), decoded through the same checks as a
/// live call. It answers only the request it was recorded for.
#[derive(Debug, Clone)]
pub struct RecordedStanzaResponse {
    body: String,
}

impl RecordedStanzaResponse {
    /// Wraps a recorded `POST /v1/analyze` response body.
    pub fn new(body: impl Into<String>) -> Self {
        Self { body: body.into() }
    }
}

impl SyntaxBackend for RecordedStanzaResponse {
    fn name(&self) -> &str {
        BACKEND
    }

    fn supports_language(&self, language: &str) -> bool {
        STANZA_LANGUAGES.contains(&language)
    }

    fn analyze(
        &self,
        request: &SyntaxRequest<'_>,
    ) -> Result<Vec<SyntaxSentence>, SyntaxBackendError> {
        decode_stanza_response(request, &self.body)
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use client::StanzaSidecar;

#[cfg(not(target_arch = "wasm32"))]
mod client {
    use std::io::{Read, Write};
    use std::net::{TcpStream, ToSocketAddrs};
    use std::time::Duration;

    use super::*;

    /// A running `sidecars/stanza` sidecar on the local machine.
    #[derive(Debug, Clone)]
    pub struct StanzaSidecar {
        host: String,
        port: u16,
        timeout: Duration,
    }

    impl Default for StanzaSidecar {
        fn default() -> Self {
            Self::new("127.0.0.1", 8765)
        }
    }

    impl StanzaSidecar {
        /// The sidecar at `host:port` with a 120 s timeout (the first request per
        /// language loads its models).
        pub fn new(host: impl Into<String>, port: u16) -> Self {
            Self {
                host: host.into(),
                port,
                timeout: Duration::from_secs(120),
            }
        }

        /// Parses `http://host:port` (no path). Only plain HTTP is spoken: the
        /// sidecar listens on the local machine.
        pub fn from_url(url: &str) -> Result<Self, SyntaxBackendError> {
            let invalid = || protocol(format!("expected http://host:port, got `{url}`"));
            let rest = url.strip_prefix("http://").ok_or_else(invalid)?;
            let rest = rest.trim_end_matches('/');
            let (host, port) = rest.rsplit_once(':').ok_or_else(invalid)?;
            let port = port.parse().map_err(|_| invalid())?;
            if host.is_empty() || host.contains('/') {
                return Err(invalid());
            }
            Ok(Self::new(host, port))
        }

        /// The sidecar at `NLP_STANZA_URL`, or [`STANZA_DEFAULT_ENDPOINT`].
        pub fn from_env() -> Result<Self, SyntaxBackendError> {
            Self::from_url(
                &std::env::var("NLP_STANZA_URL")
                    .unwrap_or_else(|_| STANZA_DEFAULT_ENDPOINT.to_string()),
            )
        }

        /// Overrides the per-request timeout.
        pub fn timeout(mut self, timeout: Duration) -> Self {
            self.timeout = timeout;
            self
        }

        /// `http://host:port`.
        pub fn endpoint(&self) -> String {
            format!("http://{}:{}", self.host, self.port)
        }

        /// `GET /v1/health`.
        pub fn health(&self) -> Result<StanzaHealth, SyntaxBackendError> {
            let (status, body) = self.exchange("GET", "/v1/health", None)?;
            if status != 200 {
                return Err(decode_error(status, &body));
            }
            serde_json::from_str(&body)
                .map_err(|error| protocol(format!("malformed health response: {error}")))
        }

        /// Like [`SyntaxBackend::analyze`], also returning the verified response
        /// body (for recording fixtures replayed with [`RecordedStanzaResponse`]).
        pub fn analyze_with_body(
            &self,
            request: &SyntaxRequest<'_>,
        ) -> Result<(Vec<SyntaxSentence>, String), SyntaxBackendError> {
            let body = serde_json::to_string(&StanzaAnalyzeRequest::new(request))
                .map_err(|error| protocol(error.to_string()))?;
            let (status, response) = self.exchange("POST", "/v1/analyze", Some(&body))?;
            if status != 200 {
                return Err(decode_error(status, &response));
            }
            let sentences = decode_stanza_response(request, &response)?;
            Ok((sentences, response))
        }

        fn unavailable(&self, message: impl std::fmt::Display) -> SyntaxBackendError {
            SyntaxBackendError::Unavailable {
                backend: BACKEND.to_string(),
                endpoint: self.endpoint(),
                message: format!("{message}; start it with `sidecars/stanza/serve`"),
            }
        }

        /// One HTTP/1.1 request with `Connection: close`; returns status and body.
        fn exchange(
            &self,
            method: &str,
            path: &str,
            body: Option<&str>,
        ) -> Result<(u16, String), SyntaxBackendError> {
            let address = (self.host.as_str(), self.port)
                .to_socket_addrs()
                .map_err(|error| self.unavailable(error))?
                .next()
                .ok_or_else(|| self.unavailable("address did not resolve"))?;
            let mut stream = TcpStream::connect_timeout(&address, self.timeout)
                .map_err(|error| self.unavailable(error))?;
            stream
                .set_read_timeout(Some(self.timeout))
                .and_then(|()| stream.set_write_timeout(Some(self.timeout)))
                .map_err(|error| self.unavailable(error))?;
            let body = body.unwrap_or("");
            let head = format!(
                "{method} {path} HTTP/1.1\r\nHost: {}:{}\r\nConnection: close\r\nAccept: application/json\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
                self.host,
                self.port,
                body.len()
            );
            stream
                .write_all(head.as_bytes())
                .and_then(|()| stream.write_all(body.as_bytes()))
                .map_err(|error| self.unavailable(error))?;
            let mut raw = Vec::new();
            stream
                .read_to_end(&mut raw)
                .map_err(|error| self.unavailable(error))?;
            parse_http_response(&raw)
        }
    }

    /// Status and body of an HTTP/1.x response with a `Content-Length` (or one
    /// delimited by the closed connection).
    pub(super) fn parse_http_response(raw: &[u8]) -> Result<(u16, String), SyntaxBackendError> {
        let split = raw
            .windows(4)
            .position(|window| window == b"\r\n\r\n")
            .ok_or_else(|| protocol("truncated HTTP response"))?;
        let head =
            std::str::from_utf8(&raw[..split]).map_err(|_| protocol("non-UTF-8 HTTP headers"))?;
        let mut lines = head.split("\r\n");
        let status = lines
            .next()
            .and_then(|line| line.split_whitespace().nth(1))
            .and_then(|code| code.parse::<u16>().ok())
            .ok_or_else(|| protocol("malformed HTTP status line"))?;
        let mut body = &raw[split + 4..];
        for line in lines {
            if let Some((name, value)) = line.split_once(':') {
                if name.trim().eq_ignore_ascii_case("content-length") {
                    let length = value
                        .trim()
                        .parse::<usize>()
                        .map_err(|_| protocol("malformed Content-Length"))?;
                    body = body
                        .get(..length)
                        .ok_or_else(|| protocol("HTTP body shorter than Content-Length"))?;
                } else if name.trim().eq_ignore_ascii_case("transfer-encoding") {
                    return Err(protocol(
                        "chunked HTTP responses are not part of the protocol",
                    ));
                }
            }
        }
        let body = String::from_utf8(body.to_vec()).map_err(|_| protocol("non-UTF-8 body"))?;
        Ok((status, body))
    }

    impl SyntaxBackend for StanzaSidecar {
        fn name(&self) -> &str {
            BACKEND
        }

        fn supports_language(&self, language: &str) -> bool {
            STANZA_LANGUAGES.contains(&language)
        }

        fn analyze(
            &self,
            request: &SyntaxRequest<'_>,
        ) -> Result<Vec<SyntaxSentence>, SyntaxBackendError> {
            if request.sentences.is_empty() {
                return Ok(Vec::new());
            }
            self.analyze_with_body(request)
                .map(|(sentences, _)| sentences)
        }
    }
}

#[cfg(test)]
mod tests;
