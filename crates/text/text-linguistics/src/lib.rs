#![doc = include_str!("../README.md")]

mod discourse;
mod entities;
mod language;
mod local_models;
mod morphology;
mod pipeline;
pub mod surface;
mod syntax;
mod tokenization;

pub use discourse::*;
pub use entities::*;
pub use language::*;
pub use local_models::*;
pub use morphology::*;
pub use pipeline::*;
pub use syntax::*;
pub use text_lexical::{rule_entities, EntityMention, EntityRuleSet};
pub use tokenization::*;

#[cfg(test)]
mod tests;
