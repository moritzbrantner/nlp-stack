#![doc = include_str!("../README.md")]

mod discourse;
mod entities;
mod language;
#[cfg(feature = "candle")]
mod lemmas;
mod local_models;
mod model_tokenizer;
mod morphology;
mod pipeline;
mod reporting;
mod routing;
mod scope;
#[cfg(feature = "stanza")]
mod stanza;
pub mod surface;
mod syntax;
mod syntax_backend;
mod temporal;
mod tokenization;
mod ud_model;

pub use discourse::*;
pub use entities::*;
pub use language::*;
#[cfg(feature = "candle")]
pub use lemmas::*;
pub use local_models::*;
pub use morphology::*;
pub use pipeline::*;
pub use reporting::*;
pub use routing::*;
pub use scope::*;
#[cfg(feature = "stanza")]
pub use stanza::*;
pub use syntax::*;
pub use syntax_backend::{
    SyntaxBackend, SyntaxBackendError, SyntaxModelVersion, SyntaxProvenance, SyntaxRequest,
    SyntaxSentence, SyntaxSentenceRequest, SyntaxToken, SyntaxWord,
};
pub use temporal::*;
pub use tokenization::*;
pub use ud_model::*;

#[cfg(test)]
mod tests;
