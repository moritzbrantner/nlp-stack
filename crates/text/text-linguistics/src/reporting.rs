//! Nested reported-speech and belief constructions read from UD dependency trees.
//!
//! "Sarah reports that John believes the river is polluted." yields two
//! constructions: *Sarah reports* with content "John believes the river is
//! polluted", and nested inside that content *John believes* with content "the
//! river is polluted". Each level keeps its cue, its grammatical source (when the
//! text names one) and its content as exact UTF-8 half-open spans.
//!
//! This is linguistic evidence of what the text reports. It does not say that a
//! source endorses the content, that anyone really holds a belief, or that the
//! content is true.

use serde::Serialize;
use text_core::{TextSpan, Token, TokenKind};

use crate::pipeline::LinguisticAnalysis;
use crate::routing::{LinguisticOperation, OperationSupport};
use crate::syntax::{DependencyRelation, DependencyTree};

/// Producer name recorded on every construction.
pub const REPORTING_PRODUCER: &str = "text-linguistics:reporting-ud-rules";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
/// What the cue verb reports, from its lemma alone.
pub enum ReportingKind {
    /// A verb of saying (`say`, `report`, `sagen`, `decir`).
    Speech,
    /// A verb of belief or opinion (`believe`, `think`, `glauben`, `creer`).
    Belief,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
/// How the content is given.
pub enum ReportingForm {
    /// A clausal complement (`that John believes ...`).
    Indirect,
    /// A quotation (`said "the river is polluted"`).
    Direct,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
/// A source-text passage: token range plus its exact text and byte span.
pub struct ReportingPassage {
    /// First token index (into [`LinguisticAnalysis::tokens`]).
    pub token_start: usize,
    /// One past the last token index.
    pub token_end: usize,
    /// The exact source text of `span`.
    pub text: String,
    /// UTF-8 half-open byte span in the source text.
    pub span: TextSpan,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
/// One reporting or belief construction.
pub struct ReportingConstruction {
    /// Position in [`ReportingEvidence::constructions`]; document order of the cue.
    pub id: usize,
    /// The construction whose content contains this one.
    pub parent: Option<usize>,
    /// Constructions directly nested in this one's content, in document order.
    pub children: Vec<usize>,
    /// Sentence index.
    pub sentence_index: usize,
    /// Language the sentence was analyzed in.
    pub language: String,
    /// Speech or belief, from the cue lemma.
    pub kind: ReportingKind,
    /// Indirect complement or direct quotation.
    pub form: ReportingForm,
    /// The cue verb token.
    pub cue: ReportingPassage,
    /// The cue's lemma used for the lexicon lookup.
    pub cue_lemma: String,
    /// The grammatical subject of the cue, when the text names one. `None` means
    /// the source does not say who reports or believes; nothing is guessed.
    pub source: Option<ReportingPassage>,
    /// The reported content, without its complementizer or quotation marks.
    pub content: ReportingPassage,
    /// The complementizer (`that`, `dass`, `que`), when present.
    pub complementizer: Option<ReportingPassage>,
    /// Producer of this construction ([`REPORTING_PRODUCER`]).
    pub producer: String,
    /// Producer of the dependency tree it was read from.
    pub dependency_producer: String,
    /// The lowest dependency-attachment confidence among the cue→content and
    /// cue→source edges. It measures the parse, not the source's commitment.
    pub confidence: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
/// Whether reporting extraction ran for one sentence, and why.
pub struct ReportingSentenceReport {
    /// Sentence index.
    pub sentence_index: usize,
    /// Coverage of the sentence. Only `Supported` sentences are read.
    pub support: OperationSupport,
    /// The dependency producer read, when extraction ran.
    pub producer: Option<String>,
    /// Why the sentence has this support level.
    pub reason: String,
    /// Constructions found in the sentence.
    pub construction_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
/// Reporting constructions of a whole analysis.
pub struct ReportingEvidence {
    /// All constructions, in cue document order. Roots have `parent == None`.
    pub constructions: Vec<ReportingConstruction>,
    /// One report per sentence, parallel to [`LinguisticAnalysis::sentences`].
    pub sentences: Vec<ReportingSentenceReport>,
}

impl ReportingEvidence {
    /// The outermost constructions.
    pub fn roots(&self) -> impl Iterator<Item = &ReportingConstruction> {
        self.constructions
            .iter()
            .filter(|item| item.parent.is_none())
    }
}

/// Reads nested reporting and belief constructions from `analysis`.
///
/// Only sentences whose dependency trees come from a UD-trained model
/// ([`OperationSupport::Supported`]) are read; heuristic trees do not mark clausal
/// complements, so those sentences are reported as unsupported rather than read.
pub fn reporting_constructions(analysis: &LinguisticAnalysis) -> ReportingEvidence {
    let mut constructions = Vec::new();
    let mut sentences = Vec::new();
    for routed in &analysis.sentence_analyses {
        let index = routed.sentence_index;
        let report = routed.operation(LinguisticOperation::DependencySyntax);
        let mut sentence_report = ReportingSentenceReport {
            sentence_index: index,
            support: OperationSupport::Unknown,
            producer: None,
            reason: String::new(),
            construction_count: 0,
        };
        let tree = analysis
            .dependencies
            .iter()
            .find(|tree| tree.sentence_index == index);
        let sentence = analysis.sentences.get(index);
        match (report, routed.language.as_deref(), tree, sentence) {
            (Some(report), Some(language), Some(tree), Some(_))
                if report.support == OperationSupport::Supported =>
            {
                let producer = report.producer.clone().unwrap_or_default();
                if lexicon(language).is_none() {
                    sentence_report.support = OperationSupport::Unsupported;
                    sentence_report.reason = format!("no reporting-verb lexicon for `{language}`");
                } else {
                    let found = sentence_constructions(
                        analysis,
                        &analysis.graph.text,
                        tree,
                        index,
                        language,
                        &producer,
                    );
                    sentence_report.support = OperationSupport::Supported;
                    sentence_report.reason = format!(
                        "clausal complements and quotations of {language} reporting and belief verbs in the {producer} dependency tree"
                    );
                    sentence_report.construction_count = found.len();
                    constructions.extend(found);
                }
                sentence_report.producer = Some(producer);
            }
            (Some(report), _, _, _) if report.support == OperationSupport::Partial => {
                sentence_report.support = OperationSupport::Unsupported;
                sentence_report.reason = format!(
                    "dependency trees come from {}, which does not mark clausal complements; reporting needs a UD dependency model",
                    report.producer.as_deref().unwrap_or("heuristics")
                );
            }
            (Some(report), _, _, _) => {
                sentence_report.support = report.support;
                sentence_report.reason = format!("no dependency tree: {}", report.reason);
            }
            (None, _, _, _) => {
                sentence_report.reason = "no dependency report for the sentence".to_string();
            }
        }
        sentences.push(sentence_report);
    }
    link_nesting(&mut constructions);
    ReportingEvidence {
        constructions,
        sentences,
    }
}

fn sentence_constructions(
    analysis: &LinguisticAnalysis,
    text: &str,
    tree: &DependencyTree,
    sentence_index: usize,
    language: &str,
    dependency_producer: &str,
) -> Vec<ReportingConstruction> {
    let tokens = &analysis.tokens;
    let mut found = Vec::new();
    let mut heads = tree
        .nodes
        .iter()
        .map(|node| node.token_index)
        .collect::<Vec<_>>();
    heads.sort_unstable();
    for cue_index in heads {
        let Some(cue_token) = tokens.get(cue_index) else {
            continue;
        };
        let cue_lemma = analysis
            .lemmas
            .iter()
            .find(|lemma| lemma.token_index == cue_index)
            .map(|lemma| lemma.value.to_lowercase())
            .unwrap_or_else(|| cue_token.text.to_lowercase());
        let Some(kind) = lexicon(language).and_then(|lexicon| lexicon.kind(&cue_lemma)) else {
            continue;
        };
        let Some(content_edge) = tree.edges.iter().find(|edge| {
            edge.head_token_index == cue_index
                && matches!(
                    edge.relation,
                    DependencyRelation::Ccomp | DependencyRelation::Parataxis
                )
                && edge.dependent_token_index > cue_index
        }) else {
            continue;
        };
        let content_head = content_edge.dependent_token_index;
        let mut confidence = content_edge.confidence;

        let mut content_tokens = subtree(tree, content_head);
        trim_punctuation(tokens, &mut content_tokens);
        let complementizer = content_tokens.first().copied().filter(|&first| {
            tree.edges.iter().any(|edge| {
                edge.dependent_token_index == first
                    && edge.head_token_index == content_head
                    && edge.relation == DependencyRelation::Mark
            })
        });
        if complementizer.is_some() {
            content_tokens.remove(0);
        }
        trim_punctuation(tokens, &mut content_tokens);
        let (Some(&start), Some(&end)) = (content_tokens.first(), content_tokens.last()) else {
            continue;
        };

        let source = tree
            .edges
            .iter()
            .find(|edge| {
                edge.head_token_index == cue_index
                    && edge.relation == DependencyRelation::Nsubj
                    && edge.subtype.as_deref() != Some("pass")
            })
            .map(|edge| {
                confidence = confidence.min(edge.confidence);
                let mut source_tokens = subtree(tree, edge.dependent_token_index);
                trim_punctuation(tokens, &mut source_tokens);
                source_tokens
            })
            .filter(|source_tokens| !source_tokens.is_empty())
            .and_then(|source_tokens| {
                passage(
                    text,
                    tokens,
                    source_tokens[0],
                    source_tokens[source_tokens.len() - 1] + 1,
                )
            });

        let (Some(content), Some(cue)) = (
            passage(text, tokens, start, end + 1),
            passage(text, tokens, cue_index, cue_index + 1),
        ) else {
            continue;
        };
        let form = if quoted(text, content.span) {
            ReportingForm::Direct
        } else {
            ReportingForm::Indirect
        };
        found.push(ReportingConstruction {
            id: 0,
            parent: None,
            children: Vec::new(),
            sentence_index,
            language: language.to_string(),
            kind,
            form,
            cue,
            cue_lemma,
            source,
            content,
            complementizer: complementizer
                .and_then(|index| passage(text, tokens, index, index + 1)),
            producer: REPORTING_PRODUCER.to_string(),
            dependency_producer: dependency_producer.to_string(),
            confidence,
        });
    }
    found
}

/// Assigns ids in cue order and links each construction to the innermost one whose
/// content contains its cue.
fn link_nesting(constructions: &mut [ReportingConstruction]) {
    constructions.sort_by_key(|item| (item.sentence_index, item.cue.token_start));
    for (id, item) in constructions.iter_mut().enumerate() {
        item.id = id;
    }
    for child in 0..constructions.len() {
        let cue = constructions[child].cue.token_start;
        let parent = (0..constructions.len())
            .filter(|&other| other != child)
            .filter(|&other| {
                let content = &constructions[other].content;
                content.token_start <= cue && cue < content.token_end
            })
            .min_by_key(|&other| {
                let content = &constructions[other].content;
                content.token_end - content.token_start
            });
        constructions[child].parent = parent;
        if let Some(parent) = parent {
            constructions[parent].children.push(child);
        }
    }
}

/// Token indices dominated by `head` (inclusive), sorted.
fn subtree(tree: &DependencyTree, head: usize) -> Vec<usize> {
    let mut members = vec![head];
    let mut frontier = vec![head];
    while let Some(node) = frontier.pop() {
        for edge in &tree.edges {
            if edge.head_token_index == node && !members.contains(&edge.dependent_token_index) {
                members.push(edge.dependent_token_index);
                frontier.push(edge.dependent_token_index);
            }
        }
    }
    members.sort_unstable();
    // The contiguous hull: a projective subtree, as in UD English/German/Spanish
    // complements, has no gaps.
    match (members.first(), members.last()) {
        (Some(&first), Some(&last)) => (first..=last).collect(),
        _ => members,
    }
}

fn trim_punctuation(tokens: &[Token], members: &mut Vec<usize>) {
    let is_punct = |index: &usize| {
        tokens
            .get(*index)
            .is_some_and(|token| token.kind == TokenKind::Punctuation)
    };
    while members.last().is_some_and(is_punct) {
        members.pop();
    }
    while members.first().is_some_and(is_punct) {
        members.remove(0);
    }
}

/// Tokens `start..end` as a source passage; `None` when their offsets do not
/// address `text`.
fn passage(text: &str, tokens: &[Token], start: usize, end: usize) -> Option<ReportingPassage> {
    let span = TextSpan::from_byte_range(
        text,
        tokens.get(start)?.span.byte_start,
        tokens.get(end.checked_sub(1)?)?.span.byte_end,
    )
    .ok()?;
    Some(ReportingPassage {
        token_start: start,
        token_end: end,
        text: text[span.byte_start..span.byte_end].to_string(),
        span,
    })
}

/// Whether the passage is enclosed in quotation marks, allowing its own final
/// punctuation inside the closing mark (`"The river is polluted."`).
fn quoted(text: &str, span: TextSpan) -> bool {
    const OPEN: &[char] = &['"', '“', '„', '«', '‘', '\'', '‹', '»'];
    const CLOSE: &[char] = &['"', '”', '“', '»', '’', '\'', '›', '«'];
    const FINAL: &[char] = &['.', '!', '?', ',', '…'];
    let before = text.get(..span.byte_start).unwrap_or_default();
    let after = text.get(span.byte_end..).unwrap_or_default();
    before.trim_end().ends_with(OPEN)
        && after
            .trim_start()
            .trim_start_matches(FINAL)
            .starts_with(CLOSE)
}

struct Lexicon {
    speech: &'static [&'static str],
    belief: &'static [&'static str],
}

impl Lexicon {
    fn kind(&self, lemma: &str) -> Option<ReportingKind> {
        if self.speech.contains(&lemma) {
            Some(ReportingKind::Speech)
        } else if self.belief.contains(&lemma) {
            Some(ReportingKind::Belief)
        } else {
            None
        }
    }
}

/// Reporting and belief verb lemmas per language. Factive verbs (`know`,
/// `realize`) are excluded: they presuppose their content instead of attributing it.
fn lexicon(language: &str) -> Option<&'static Lexicon> {
    const EN: Lexicon = Lexicon {
        speech: &[
            "say", "report", "tell", "claim", "state", "announce", "write", "insist", "argue",
            "explain", "add", "reply", "mention", "assert", "declare", "confirm", "deny",
            "suggest", "warn", "note", "testify", "admit", "allege", "maintain", "answer",
        ],
        belief: &[
            "believe", "think", "suppose", "assume", "suspect", "doubt", "feel", "expect", "hope",
            "fear", "imagine", "guess", "reckon", "consider",
        ],
    };
    const DE: Lexicon = Lexicon {
        speech: &[
            "sagen",
            "berichten",
            "erzählen",
            "behaupten",
            "erklären",
            "schreiben",
            "melden",
            "mitteilen",
            "antworten",
            "betonen",
            "versichern",
            "bestätigen",
            "bestreiten",
            "verkünden",
            "warnen",
        ],
        belief: &[
            "glauben",
            "denken",
            "meinen",
            "vermuten",
            "annehmen",
            "bezweifeln",
            "hoffen",
            "fürchten",
            "befürchten",
            "finden",
            "erwarten",
        ],
    };
    const ES: Lexicon = Lexicon {
        speech: &[
            "decir",
            "informar",
            "contar",
            "afirmar",
            "declarar",
            "explicar",
            "escribir",
            "asegurar",
            "añadir",
            "responder",
            "anunciar",
            "negar",
            "confirmar",
            "advertir",
        ],
        belief: &[
            "creer",
            "pensar",
            "suponer",
            "opinar",
            "sospechar",
            "dudar",
            "esperar",
            "temer",
            "considerar",
        ],
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
    use crate::morphology::Lemma;
    use crate::routing::OperationReport;
    use crate::syntax::{DependencyEdge, DependencyNode};
    use crate::{analyze_text, LinguisticAnalysisOptions};

    /// The heuristic analysis of `text` with a hand-written UD tree standing in for
    /// the model parse. `arcs` are `(dependent, head, deprel)` by token text (first
    /// occurrence); `head == ""` is the root. `lemmas` are `(token, lemma)`.
    fn fixture(
        text: &str,
        arcs: &[(&str, &str, &str)],
        lemmas: &[(&str, &str)],
    ) -> LinguisticAnalysis {
        let mut analysis = analyze_text(text, &LinguisticAnalysisOptions::default()).unwrap();
        let index = |word: &str| {
            analysis
                .tokens
                .iter()
                .position(|token| token.text == word)
                .unwrap_or_else(|| panic!("no token {word:?}"))
        };
        let mut tree = DependencyTree {
            sentence_index: 0,
            root_token_index: None,
            nodes: Vec::new(),
            edges: Vec::new(),
        };
        for &(dependent, head, deprel) in arcs {
            let (relation, subtype) = DependencyRelation::parse_ud_deprel(deprel);
            let dependent = index(dependent);
            let head = (!head.is_empty()).then(|| index(head));
            if head.is_none() {
                tree.root_token_index = Some(dependent);
            }
            tree.nodes.push(DependencyNode {
                token_index: dependent,
                head_token_index: head,
                relation,
                subtype: subtype.clone(),
            });
            if let Some(head) = head {
                tree.edges.push(DependencyEdge {
                    head_token_index: head,
                    dependent_token_index: dependent,
                    relation,
                    subtype,
                    confidence: 0.9,
                });
            }
        }
        analysis.lemmas = lemmas
            .iter()
            .map(|&(word, lemma)| Lemma {
                token_index: index(word),
                value: lemma.to_string(),
                language: Some("en".to_string()),
                confidence: 0.9,
            })
            .collect();
        analysis.dependencies = vec![tree];
        let routed = &mut analysis.sentence_analyses[0];
        routed.language = Some("en".to_string());
        for report in &mut routed.operations {
            if report.operation == LinguisticOperation::DependencySyntax {
                *report = OperationReport::new(
                    LinguisticOperation::DependencySyntax,
                    OperationSupport::Supported,
                    Some("ud-model:fixture".to_string()),
                    "fixture",
                );
            }
        }
        analysis
    }

    fn nested_fixture() -> (&'static str, LinguisticAnalysis) {
        let text = "Zoë reports that Jürgen believes the café is closed.";
        let analysis = fixture(
            text,
            &[
                ("Zoë", "reports", "nsubj"),
                ("reports", "", "root"),
                ("that", "believes", "mark"),
                ("Jürgen", "believes", "nsubj"),
                ("believes", "reports", "ccomp"),
                ("the", "café", "det"),
                ("café", "closed", "nsubj"),
                ("is", "closed", "cop"),
                ("closed", "believes", "ccomp"),
                (".", "reports", "punct"),
            ],
            &[
                ("reports", "report"),
                ("believes", "believe"),
                ("is", "be"),
                ("closed", "closed"),
            ],
        );
        (text, analysis)
    }

    #[test]
    fn nests_belief_inside_report_with_exact_unicode_spans() {
        let (text, analysis) = nested_fixture();
        let evidence = reporting_constructions(&analysis);
        assert_eq!(evidence.sentences[0].support, OperationSupport::Supported);
        assert_eq!(evidence.sentences[0].construction_count, 2);
        let [outer, inner] = evidence.constructions.as_slice() else {
            panic!("{:?}", evidence.constructions);
        };
        assert_eq!(
            (outer.kind, inner.kind),
            (ReportingKind::Speech, ReportingKind::Belief)
        );
        assert_eq!((outer.parent, &outer.children), (None, &vec![1]));
        assert_eq!(inner.parent, Some(0));
        assert_eq!(outer.source.as_ref().unwrap().text, "Zoë");
        assert_eq!(outer.content.text, "Jürgen believes the café is closed");
        assert_eq!(outer.complementizer.as_ref().unwrap().text, "that");
        assert_eq!(inner.source.as_ref().unwrap().text, "Jürgen");
        assert_eq!(inner.content.text, "the café is closed");
        assert_eq!(inner.form, ReportingForm::Indirect);
        assert_eq!(inner.dependency_producer, "ud-model:fixture");
        for item in &evidence.constructions {
            for passage in [Some(&item.cue), item.source.as_ref(), Some(&item.content)]
                .into_iter()
                .flatten()
            {
                assert_eq!(
                    &text[passage.span.byte_start..passage.span.byte_end],
                    passage.text
                );
            }
        }
        // The inner content lies inside the outer content.
        assert!(outer.content.span.byte_start <= inner.cue.span.byte_start);
        assert!(inner.content.span.byte_end <= outer.content.span.byte_end);
        assert_eq!(evidence.roots().count(), 1);
    }

    #[test]
    fn repeat_runs_give_identical_evidence() {
        let (_, analysis) = nested_fixture();
        assert_eq!(
            reporting_constructions(&analysis),
            reporting_constructions(&analysis)
        );
    }

    #[test]
    fn plain_statement_has_no_construction() {
        let analysis = fixture(
            "The river is polluted.",
            &[
                ("The", "river", "det"),
                ("river", "polluted", "nsubj"),
                ("is", "polluted", "cop"),
                ("polluted", "", "root"),
                (".", "polluted", "punct"),
            ],
            &[("is", "be")],
        );
        let evidence = reporting_constructions(&analysis);
        assert!(evidence.constructions.is_empty());
        assert_eq!(evidence.sentences[0].support, OperationSupport::Supported);
    }

    #[test]
    fn quoted_speech_is_speech_not_belief() {
        let analysis = fixture(
            "John said, \"The river is polluted.\"",
            &[
                ("John", "said", "nsubj"),
                ("said", "", "root"),
                (",", "said", "punct"),
                ("The", "river", "det"),
                ("river", "polluted", "nsubj"),
                ("is", "polluted", "cop"),
                ("polluted", "said", "ccomp"),
            ],
            &[("said", "say")],
        );
        let evidence = reporting_constructions(&analysis);
        let [only] = evidence.constructions.as_slice() else {
            panic!("{:?}", evidence.constructions);
        };
        assert_eq!(only.kind, ReportingKind::Speech);
        assert_eq!(only.form, ReportingForm::Direct);
        assert_eq!(only.content.text, "The river is polluted");
    }

    #[test]
    fn passive_report_names_no_source() {
        let analysis = fixture(
            "It is believed that the river is polluted.",
            &[
                ("It", "believed", "nsubj:pass"),
                ("is", "believed", "aux:pass"),
                ("believed", "", "root"),
                ("that", "polluted", "mark"),
                ("the", "river", "det"),
                ("river", "polluted", "nsubj"),
                ("polluted", "believed", "ccomp"),
                (".", "believed", "punct"),
            ],
            &[("believed", "believe")],
        );
        let evidence = reporting_constructions(&analysis);
        let [only] = evidence.constructions.as_slice() else {
            panic!("{:?}", evidence.constructions);
        };
        assert_eq!(only.kind, ReportingKind::Belief);
        assert_eq!(only.source, None);
    }

    #[test]
    fn heuristic_dependencies_report_unsupported_without_constructions() {
        let text = "Sarah reports that John believes the river is polluted.";
        let analysis = analyze_text(text, &LinguisticAnalysisOptions::default()).unwrap();
        let evidence = reporting_constructions(&analysis);
        assert!(evidence.constructions.is_empty());
        assert_eq!(evidence.sentences.len(), 1);
        assert_ne!(evidence.sentences[0].support, OperationSupport::Supported);
        assert!(!evidence.sentences[0].reason.is_empty());
    }
}
