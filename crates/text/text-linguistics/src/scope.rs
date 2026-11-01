//! Negation, uncertainty and hypothetical operators with their dependency scope.
//!
//! "Sarah says John does not believe the river is polluted." yields one negation
//! whose cue `not` modifies `believe`: its scope is "John does not believe the
//! river is polluted", it targets John's belief construction and lies inside
//! Sarah's report. "Sarah does not say …" instead targets Sarah's report. Each
//! operator keeps its cue, the construction it modifies and its scope as exact
//! UTF-8 half-open spans, and links to the reporting constructions of
//! [`crate::reporting_constructions`] and to the operators it is nested in.
//!
//! This is linguistic evidence of what the text marks. A negated belief is not a
//! false proposition, an uncertain event did not happen, and a conditional's
//! antecedent and consequent are not observed events. `confidence` measures the
//! dependency parse, never the text's own hedging.

use serde::Serialize;

use crate::pipeline::LinguisticAnalysis;
use crate::reporting::{
    passage, subtree, token_lemma, trim_punctuation, ud_sentence, ReportingEvidence, SourcePassage,
};
use crate::syntax::{DependencyRelation, DependencyTree};
use crate::OperationSupport;

/// Producer name recorded on every operator.
pub const SCOPE_PRODUCER: &str = "text-linguistics:scope-ud-rules";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
/// What the cue marks.
pub enum ScopeOperatorKind {
    /// Linguistic negation (`not`, `never`, `nicht`, `kein`, `no`, `nunca`).
    Negation,
    /// Uncertainty the text itself expresses: a possibility modal (`might`,
    /// `könnte`, `podría`) or a hedging adverb (`perhaps`, `vielleicht`, `quizás`).
    Uncertainty,
    /// A conditional antecedent (`if`, `falls`, `wenn`, `si`).
    Hypothetical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
/// Whether the parse gives the operator a scope.
pub enum ScopeResolution {
    /// The cue attaches by a relation that determines its scope.
    Resolved,
    /// The cue was found but its attachment does not determine a scope; `head`,
    /// `scope` and `consequent` are `None` instead of guessed.
    Unresolved,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
/// One negation, uncertainty or hypothetical operator.
pub struct ScopeOperator {
    /// Position in [`ScopeEvidence::operators`]; document order of the cue.
    pub id: usize,
    /// The innermost operator whose scope strictly contains this one.
    pub parent: Option<usize>,
    /// Operators directly nested in this one's scope, in document order.
    pub children: Vec<usize>,
    /// Sentence index.
    pub sentence_index: usize,
    /// Language the sentence was analyzed in.
    pub language: String,
    /// Negation, uncertainty or hypothetical.
    pub kind: ScopeOperatorKind,
    /// The cue token, in its original wording.
    pub cue: SourcePassage,
    /// The cue's lemma used for the lexicon lookup.
    pub cue_lemma: String,
    /// The cue's UD relation to its head (`advmod`, `aux`, `det`, `mark`).
    pub cue_relation: String,
    /// Whether the parse determines a scope.
    pub resolution: ScopeResolution,
    /// Why the scope is unresolved; `None` when resolved.
    pub unresolved_reason: Option<String>,
    /// The word the cue modifies: the negated or hedged predicate or constituent,
    /// or a conditional's antecedent predicate.
    pub head: Option<SourcePassage>,
    /// The affected construction: the head's subtree. For a conditional, the
    /// antecedent clause without its `if`.
    pub scope: Option<SourcePassage>,
    /// A conditional's consequent clause (the clause the antecedent attaches to).
    pub consequent: Option<SourcePassage>,
    /// The reporting construction ([`ReportingEvidence::constructions`] id) whose
    /// cue the operator modifies (`does not say`), or whose cue heads a
    /// conditional's consequent.
    pub reporting_target: Option<usize>,
    /// The innermost reporting construction whose content contains the cue
    /// (`Sarah says [John does not believe …]`).
    pub reported_within: Option<usize>,
    /// Producer of this operator ([`SCOPE_PRODUCER`]).
    pub producer: String,
    /// Producer of the dependency tree it was read from.
    pub dependency_producer: String,
    /// The lowest dependency-attachment confidence of the edges read. It measures
    /// the parse, not the text's certainty; `None` when no edge was read.
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
/// Whether scope extraction ran for one sentence, and why.
pub struct ScopeSentenceReport {
    /// Sentence index.
    pub sentence_index: usize,
    /// Coverage of the sentence. Only `Supported` sentences are read.
    pub support: OperationSupport,
    /// The dependency producer read, when extraction ran.
    pub producer: Option<String>,
    /// Why the sentence has this support level.
    pub reason: String,
    /// Operators found in the sentence.
    pub operator_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
/// Scope operators of a whole analysis.
pub struct ScopeEvidence {
    /// All operators, in cue document order. Roots have `parent == None`.
    pub operators: Vec<ScopeOperator>,
    /// One report per sentence, parallel to [`LinguisticAnalysis::sentences`].
    pub sentences: Vec<ScopeSentenceReport>,
}

/// Reads negation, uncertainty and hypothetical operators from `analysis`, linked
/// to the reporting constructions in `reporting` (from
/// [`crate::reporting_constructions`] of the same analysis).
///
/// Only sentences whose dependency trees come from a UD-trained model are read;
/// heuristic trees do not attach cues to the words they modify, so those
/// sentences are reported as unsupported.
pub fn scope_operators(
    analysis: &LinguisticAnalysis,
    reporting: &ReportingEvidence,
) -> ScopeEvidence {
    let mut operators = Vec::new();
    let mut sentences = Vec::new();
    for routed in &analysis.sentence_analyses {
        let index = routed.sentence_index;
        let mut report = ScopeSentenceReport {
            sentence_index: index,
            support: OperationSupport::Unknown,
            producer: None,
            reason: String::new(),
            operator_count: 0,
        };
        match ud_sentence(
            analysis,
            routed,
            "does not attach negation, modal and conditional cues; scope needs a UD dependency model",
        ) {
            Ok(parsed) => {
                if lexicon(parsed.language).is_none() {
                    report.support = OperationSupport::Unsupported;
                    report.reason = format!("no scope-cue lexicon for `{}`", parsed.language);
                } else {
                    let found = sentence_operators(
                        analysis,
                        parsed.tree,
                        index,
                        parsed.language,
                        &parsed.producer,
                        reporting,
                    );
                    report.support = OperationSupport::Supported;
                    report.reason = format!(
                        "negation, possibility and conditional cues of {} and their attachment in the {} dependency tree",
                        parsed.language, parsed.producer
                    );
                    report.operator_count = found.len();
                    operators.extend(found);
                }
                report.producer = Some(parsed.producer);
            }
            Err((support, reason)) => {
                report.support = support;
                report.reason = reason;
            }
        }
        sentences.push(report);
    }
    link_nesting(&mut operators);
    ScopeEvidence {
        operators,
        sentences,
    }
}

fn sentence_operators(
    analysis: &LinguisticAnalysis,
    tree: &DependencyTree,
    sentence_index: usize,
    language: &str,
    dependency_producer: &str,
    reporting: &ReportingEvidence,
) -> Vec<ScopeOperator> {
    let text = &analysis.graph.text;
    let tokens = &analysis.tokens;
    let Some(lexicon) = lexicon(language) else {
        return Vec::new();
    };
    let edge_to = |dependent: usize| {
        tree.edges
            .iter()
            .find(|edge| edge.dependent_token_index == dependent)
    };
    let span_of = |members: &[usize]| match (members.first(), members.last()) {
        (Some(&first), Some(&last)) => passage(text, tokens, first, last + 1),
        _ => None,
    };
    let mut nodes = tree.nodes.iter().collect::<Vec<_>>();
    nodes.sort_by_key(|node| node.token_index);

    let mut found = Vec::new();
    for node in nodes {
        let cue_index = node.token_index;
        let Some(token) = tokens.get(cue_index) else {
            continue;
        };
        let lemma = token_lemma(analysis, cue_index);
        let form = token.text.to_lowercase();
        let Some(kind) = lexicon.kind(&lemma, &form, node.relation) else {
            continue;
        };
        let Some(cue) = passage(text, tokens, cue_index, cue_index + 1) else {
            continue;
        };
        let edge = edge_to(cue_index);
        // `if`/`si` introducing a complement (`I wonder if …`) asks; it is not a condition.
        if kind == ScopeOperatorKind::Hypothetical
            && edge
                .and_then(|edge| edge_to(edge.head_token_index))
                .is_some_and(|clause| {
                    matches!(
                        clause.relation,
                        DependencyRelation::Ccomp
                            | DependencyRelation::Csubj
                            | DependencyRelation::Xcomp
                    )
                })
        {
            continue;
        }
        let mut item = ScopeOperator {
            id: 0,
            parent: None,
            children: Vec::new(),
            sentence_index,
            language: language.to_string(),
            kind,
            cue,
            cue_lemma: lemma,
            cue_relation: deprel(node.relation, node.subtype.as_deref()),
            resolution: ScopeResolution::Unresolved,
            unresolved_reason: None,
            head: None,
            scope: None,
            consequent: None,
            reporting_target: None,
            reported_within: innermost_report(reporting, sentence_index, cue_index),
            producer: SCOPE_PRODUCER.to_string(),
            dependency_producer: dependency_producer.to_string(),
            confidence: edge.map(|edge| edge.confidence),
        };
        let Some(edge) = edge else {
            item.unresolved_reason =
                Some("the cue is the root of the parse and modifies nothing".to_string());
            found.push(item);
            continue;
        };
        let head = edge.head_token_index;
        let mut target = head;
        let resolved = match kind {
            ScopeOperatorKind::Negation | ScopeOperatorKind::Uncertainty => {
                if matches!(
                    node.relation,
                    DependencyRelation::Advmod | DependencyRelation::Det | DependencyRelation::Aux
                ) {
                    let mut members = subtree(tree, head);
                    trim_punctuation(tokens, &mut members);
                    // A subordinator (`if`, `that`) introduces the clause from outside it.
                    while members.first().is_some_and(|&first| {
                        first != cue_index
                            && tree.edges.iter().any(|edge| {
                                edge.dependent_token_index == first
                                    && edge.head_token_index == head
                                    && edge.relation == DependencyRelation::Mark
                            })
                    }) {
                        members.remove(0);
                    }
                    trim_punctuation(tokens, &mut members);
                    item.scope = span_of(&members);
                    true
                } else {
                    item.unresolved_reason = Some(format!(
                        "the cue attaches as `{}`, which does not determine what it modifies",
                        item.cue_relation
                    ));
                    false
                }
            }
            ScopeOperatorKind::Hypothetical => {
                let antecedent_edge = edge_to(head)
                    .filter(|antecedent| antecedent.relation == DependencyRelation::Advcl);
                match antecedent_edge {
                    Some(antecedent_edge) => {
                        let mut antecedent = subtree(tree, head);
                        antecedent.retain(|&index| index != cue_index);
                        trim_punctuation(tokens, &mut antecedent);
                        item.scope = span_of(&antecedent);
                        let consequent_head = antecedent_edge.head_token_index;
                        let consequent =
                            clause_without(tree, tokens, consequent_head, &antecedent, cue_index);
                        item.consequent = span_of(&consequent);
                        item.confidence = Some(edge.confidence.min(antecedent_edge.confidence));
                        target = consequent_head;
                        true
                    }
                    None => {
                        item.unresolved_reason = Some(
                            "the conditional's clause is not attached as an adverbial clause (`advcl`), so its consequent is not determined"
                                .to_string(),
                        );
                        false
                    }
                }
            }
        };
        if resolved && item.scope.is_some() {
            item.resolution = ScopeResolution::Resolved;
            item.head = passage(text, tokens, head, head + 1);
            item.reporting_target = reporting
                .constructions
                .iter()
                .find(|construction| {
                    construction.sentence_index == sentence_index
                        && construction.cue.token_start == target
                })
                .map(|construction| construction.id);
        } else if resolved {
            item.scope = None;
            item.consequent = None;
            item.unresolved_reason = Some("the scope has no source text".to_string());
        }
        found.push(item);
    }
    found
}

/// The contiguous run of `head`'s clause around `head`, without the antecedent,
/// the cue and punctuation.
fn clause_without(
    tree: &DependencyTree,
    tokens: &[text_core::Token],
    head: usize,
    antecedent: &[usize],
    cue_index: usize,
) -> Vec<usize> {
    let mut members = subtree(tree, head);
    members.retain(|index| *index != cue_index && !antecedent.contains(index));
    let is_punct = |index: usize| {
        tokens
            .get(index)
            .is_some_and(|token| token.kind == text_core::TokenKind::Punctuation)
    };
    let Some(position) = members.iter().position(|&index| index == head) else {
        return Vec::new();
    };
    let mut start = position;
    while start > 0 && members[start - 1] + 1 == members[start] && !is_punct(members[start - 1]) {
        start -= 1;
    }
    let mut end = position;
    while end + 1 < members.len()
        && members[end] + 1 == members[end + 1]
        && !is_punct(members[end + 1])
    {
        end += 1;
    }
    members[start..=end].to_vec()
}

fn innermost_report(
    reporting: &ReportingEvidence,
    sentence_index: usize,
    cue: usize,
) -> Option<usize> {
    reporting
        .constructions
        .iter()
        .filter(|construction| {
            construction.sentence_index == sentence_index
                && construction.content.token_start <= cue
                && cue < construction.content.token_end
        })
        .min_by_key(|construction| {
            construction.content.token_end - construction.content.token_start
        })
        .map(|construction| construction.id)
}

fn deprel(relation: DependencyRelation, subtype: Option<&str>) -> String {
    match subtype {
        Some(subtype) => format!("{}:{subtype}", relation.as_ud()),
        None => relation.as_ud().to_string(),
    }
}

/// Assigns ids in cue order and links each operator to the innermost resolved
/// operator whose scope contains its cue while its own scope does not contain that
/// operator's cue. Two operators whose scopes contain each other's cues
/// (`might not leave`) stay siblings: the parse does not order them.
fn link_nesting(operators: &mut [ScopeOperator]) {
    operators.sort_by_key(|item| (item.sentence_index, item.cue.token_start));
    for (id, item) in operators.iter_mut().enumerate() {
        item.id = id;
    }
    let width = |passage: &SourcePassage| passage.token_end - passage.token_start;
    for child in 0..operators.len() {
        let cue = operators[child].cue.token_start;
        let own = operators[child].scope.clone();
        let parent = (0..operators.len())
            .filter(|&other| other != child)
            .filter(|&other| operators[other].sentence_index == operators[child].sentence_index)
            .filter_map(|other| {
                let scope = operators[other].scope.as_ref()?;
                let other_cue = operators[other].cue.token_start;
                let mutual = own
                    .as_ref()
                    .is_some_and(|own| own.token_start <= other_cue && other_cue < own.token_end);
                (scope.token_start <= cue && cue < scope.token_end && !mutual)
                    .then_some((other, width(scope)))
            })
            .min_by_key(|&(_, width)| width)
            .map(|(other, _)| other);
        operators[child].parent = parent;
        if let Some(parent) = parent {
            operators[parent].children.push(child);
        }
    }
}

struct Lexicon {
    negation: &'static [&'static str],
    /// Possibility modals, read only as `aux`.
    modals: &'static [&'static str],
    /// Hedging adverbs, read only as `advmod`.
    hedges: &'static [&'static str],
    /// Conditional subordinators, read only as `mark`.
    conditionals: &'static [&'static str],
}

impl Lexicon {
    fn kind(
        &self,
        lemma: &str,
        form: &str,
        relation: DependencyRelation,
    ) -> Option<ScopeOperatorKind> {
        let has = |list: &[&str]| list.contains(&lemma) || list.contains(&form);
        match relation {
            // An interjection `no` answers; it negates nothing in the sentence.
            DependencyRelation::Discourse => None,
            _ if has(self.negation) => Some(ScopeOperatorKind::Negation),
            DependencyRelation::Aux if has(self.modals) => Some(ScopeOperatorKind::Uncertainty),
            DependencyRelation::Advmod if has(self.hedges) => Some(ScopeOperatorKind::Uncertainty),
            DependencyRelation::Mark if has(self.conditionals) => {
                Some(ScopeOperatorKind::Hypothetical)
            }
            _ => None,
        }
    }
}

/// Cue words per language, matched against lemma or lowercased form.
fn lexicon(language: &str) -> Option<&'static Lexicon> {
    const EN: Lexicon = Lexicon {
        negation: &["not", "n't", "never", "no"],
        modals: &["might", "may", "could"],
        hedges: &[
            "perhaps",
            "maybe",
            "probably",
            "possibly",
            "likely",
            "apparently",
            "allegedly",
            "supposedly",
            "presumably",
        ],
        conditionals: &["if"],
    };
    const DE: Lexicon = Lexicon {
        negation: &[
            "nicht", "nie", "niemals", "kein", "keine", "keinen", "keinem", "keiner", "keines",
        ],
        modals: &["könnte", "könnten", "könntest", "dürfte", "dürften"],
        hedges: &[
            "vielleicht",
            "wahrscheinlich",
            "möglicherweise",
            "vermutlich",
            "eventuell",
            "angeblich",
        ],
        conditionals: &["falls", "wenn", "sofern"],
    };
    const ES: Lexicon = Lexicon {
        negation: &["no", "nunca", "jamás", "tampoco"],
        modals: &["podría", "podrían", "podrías", "podríamos"],
        hedges: &[
            "quizá",
            "quizás",
            "probablemente",
            "posiblemente",
            "acaso",
            "supuestamente",
        ],
        conditionals: &["si"],
    };
    match language {
        "en" => Some(&EN),
        "de" => Some(&DE),
        "es" => Some(&ES),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reporting::tests::fixture;
    use crate::{analyze_text, reporting_constructions, LinguisticAnalysisOptions};

    fn evidence(analysis: &LinguisticAnalysis) -> (ReportingEvidence, ScopeEvidence) {
        let reporting = reporting_constructions(analysis);
        let scope = scope_operators(analysis, &reporting);
        (reporting, scope)
    }

    fn assert_exact(text: &str, scope: &ScopeEvidence) {
        for item in &scope.operators {
            for passage in [
                Some(&item.cue),
                item.head.as_ref(),
                item.scope.as_ref(),
                item.consequent.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                assert_eq!(
                    &text[passage.span.byte_start..passage.span.byte_end],
                    passage.text
                );
            }
        }
    }

    #[test]
    fn negated_belief_inside_report_targets_the_belief() {
        let text = "Zoë says Jürgen does not believe the café is closed.";
        let analysis = fixture(
            text,
            &[
                ("Zoë", "says", "nsubj"),
                ("says", "", "root"),
                ("Jürgen", "believe", "nsubj"),
                ("does", "believe", "aux"),
                ("not", "believe", "advmod"),
                ("believe", "says", "ccomp"),
                ("the", "café", "det"),
                ("café", "closed", "nsubj"),
                ("is", "closed", "cop"),
                ("closed", "believe", "ccomp"),
                (".", "says", "punct"),
            ],
            &[("says", "say"), ("believe", "believe"), ("does", "do")],
        );
        let (reporting, scope) = evidence(&analysis);
        assert_eq!(scope.sentences[0].support, OperationSupport::Supported);
        let [negation] = scope.operators.as_slice() else {
            panic!("{:?}", scope.operators);
        };
        assert_eq!(negation.kind, ScopeOperatorKind::Negation);
        assert_eq!(negation.resolution, ScopeResolution::Resolved);
        assert_eq!(negation.cue.text, "not");
        assert_eq!(negation.head.as_ref().unwrap().text, "believe");
        assert_eq!(
            negation.scope.as_ref().unwrap().text,
            "Jürgen does not believe the café is closed"
        );
        let belief = &reporting.constructions[negation.reporting_target.unwrap()];
        assert_eq!(belief.cue.text, "believe");
        let report = &reporting.constructions[negation.reported_within.unwrap()];
        assert_eq!(report.cue.text, "says");
        assert_exact(text, &scope);
    }

    #[test]
    fn negated_report_targets_the_report_not_the_belief() {
        let text = "Zoë does not say Jürgen believes the café is closed.";
        let analysis = fixture(
            text,
            &[
                ("Zoë", "say", "nsubj"),
                ("does", "say", "aux"),
                ("not", "say", "advmod"),
                ("say", "", "root"),
                ("Jürgen", "believes", "nsubj"),
                ("believes", "say", "ccomp"),
                ("the", "café", "det"),
                ("café", "closed", "nsubj"),
                ("is", "closed", "cop"),
                ("closed", "believes", "ccomp"),
                (".", "say", "punct"),
            ],
            &[("say", "say"), ("believes", "believe"), ("does", "do")],
        );
        let (reporting, scope) = evidence(&analysis);
        let [negation] = scope.operators.as_slice() else {
            panic!("{:?}", scope.operators);
        };
        assert_eq!(negation.head.as_ref().unwrap().text, "say");
        assert_eq!(
            reporting.constructions[negation.reporting_target.unwrap()]
                .cue
                .text,
            "say"
        );
        assert_eq!(negation.reported_within, None);
        // The belief is still reported, unnegated, inside the negated report.
        let belief = reporting
            .constructions
            .iter()
            .find(|item| item.cue.text == "believes")
            .unwrap();
        assert_ne!(negation.reporting_target, Some(belief.id));
        assert_exact(text, &scope);
    }

    #[test]
    fn modal_marks_uncertainty_not_occurrence() {
        let analysis = fixture(
            "John might leave.",
            &[
                ("John", "leave", "nsubj"),
                ("might", "leave", "aux"),
                ("leave", "", "root"),
                (".", "leave", "punct"),
            ],
            &[("might", "might"), ("leave", "leave")],
        );
        let (_, scope) = evidence(&analysis);
        let [only] = scope.operators.as_slice() else {
            panic!("{:?}", scope.operators);
        };
        assert_eq!(only.kind, ScopeOperatorKind::Uncertainty);
        assert_eq!(only.scope.as_ref().unwrap().text, "John might leave");
        assert_eq!(only.cue_relation, "aux");
    }

    #[test]
    fn conditional_keeps_antecedent_and_consequent_with_nested_negation() {
        let text = "If Zoë doesn't leave, Jürgen will call.";
        // The heuristic tokenizer's split of "doesn't" decides the arcs below.
        let words = analyze_text(text, &LinguisticAnalysisOptions::default())
            .unwrap()
            .tokens
            .into_iter()
            .map(|token| token.text)
            .collect::<Vec<_>>();
        let (does, not) = if words.iter().any(|word| word == "n't") {
            ("does", "n't")
        } else {
            ("doesn't", "doesn't")
        };
        let mut arcs = vec![
            ("If", "leave", "mark"),
            ("Zoë", "leave", "nsubj"),
            ("leave", "call", "advcl"),
            (",", "leave", "punct"),
            ("Jürgen", "call", "nsubj"),
            ("will", "call", "aux"),
            ("call", "", "root"),
            (".", "call", "punct"),
        ];
        if does != not {
            arcs.push((does, "leave", "aux"));
        }
        arcs.push((not, "leave", "advmod"));
        let analysis = fixture(text, &arcs, &[(not, "not")]);
        let (_, scope) = evidence(&analysis);
        let [conditional, negation] = scope.operators.as_slice() else {
            panic!("{:?}", scope.operators);
        };
        assert_eq!(conditional.kind, ScopeOperatorKind::Hypothetical);
        assert_eq!(conditional.cue.text, "If");
        assert_eq!(
            conditional.scope.as_ref().unwrap().text,
            "Zoë doesn't leave"
        );
        assert_eq!(
            conditional.consequent.as_ref().unwrap().text,
            "Jürgen will call"
        );
        assert_eq!(negation.kind, ScopeOperatorKind::Negation);
        assert_eq!(negation.parent, Some(conditional.id));
        assert_eq!(conditional.children, vec![negation.id]);
        assert_exact(text, &scope);
    }

    #[test]
    fn interrogative_if_is_not_hypothetical_and_odd_attachment_is_unresolved() {
        let analysis = fixture(
            "I wonder if John left.",
            &[
                ("I", "wonder", "nsubj"),
                ("wonder", "", "root"),
                ("if", "left", "mark"),
                ("John", "left", "nsubj"),
                ("left", "wonder", "ccomp"),
                (".", "wonder", "punct"),
            ],
            &[("left", "leave")],
        );
        assert!(evidence(&analysis).1.operators.is_empty());

        let analysis = fixture(
            "Not John left.",
            &[
                ("Not", "", "root"),
                ("John", "left", "nsubj"),
                ("left", "Not", "conj"),
                (".", "Not", "punct"),
            ],
            &[("Not", "not"), ("left", "leave")],
        );
        let (_, scope) = evidence(&analysis);
        let [only] = scope.operators.as_slice() else {
            panic!("{:?}", scope.operators);
        };
        assert_eq!(only.resolution, ScopeResolution::Unresolved);
        assert_eq!((&only.head, &only.scope), (&None, &None));
        assert!(only.unresolved_reason.is_some());
    }

    #[test]
    fn repeat_runs_give_identical_evidence() {
        let analysis = fixture(
            "John might leave.",
            &[
                ("John", "leave", "nsubj"),
                ("might", "leave", "aux"),
                ("leave", "", "root"),
            ],
            &[],
        );
        assert_eq!(evidence(&analysis), evidence(&analysis));
    }

    #[test]
    fn heuristic_dependencies_report_unsupported_without_operators() {
        let analysis = analyze_text(
            "John does not believe it.",
            &LinguisticAnalysisOptions::default(),
        )
        .unwrap();
        let (_, scope) = evidence(&analysis);
        assert!(scope.operators.is_empty());
        assert_ne!(scope.sentences[0].support, OperationSupport::Supported);
        assert!(!scope.sentences[0].reason.is_empty());
    }
}
