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
pub mod surface;
mod syntax;
mod temporal;
mod tokenization;
mod ud_morphology;

pub use discourse::*;
pub use entities::*;
pub use language::*;
#[cfg(feature = "candle")]
pub use lemmas::*;
pub use local_models::*;
pub use morphology::*;
pub use pipeline::*;
pub use syntax::*;
pub use temporal::*;
pub use text_lexical::{rule_entities, EntityMention, EntityRuleSet};
pub use tokenization::*;
pub use ud_morphology::*;

#[cfg(test)]
mod tests;
