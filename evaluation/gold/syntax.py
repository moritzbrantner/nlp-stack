"""UD syntax scoring in the style of the CoNLL 2018 shared-task scorer.

System words are aligned to gold words by their character spans over the sentence
text without whitespace; inside multi-word tokens they are aligned by the longest
common subsequence of lowercased forms. Every metric is an F1 over gold and system
words (precision over system words, recall over gold words), so tokenization
errors cost both. ``alignedAccuracy`` divides by aligned words only, the number
model cards often call "aligned accuracy".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from evaluation.gold.conllu import Sentence, Word

# Universal features compared by the CoNLL 2018 scorer for UFeats.
UNIVERSAL_FEATURES = frozenset(
    "PronType NumType Poss Reflex Foreign Abbr Typo Gender Animacy NounClass Number Case "
    "Definite Degree VerbForm Mood Tense Aspect Voice Evident Polarity Person Polite "
    "Clusivity".split()
)

METRICS = ("tokens", "words", "upos", "ufeats", "lemma", "uas", "las")


class SyntaxAlignmentError(ValueError):
    pass


@dataclass(frozen=True)
class _SpanWord:
    word: Word
    start: int
    end: int
    multiword: bool


def _span_words(sentence: Sentence) -> Tuple[List[_SpanWord], List[Tuple[int, int]], str]:
    words: List[_SpanWord] = []
    token_spans: List[Tuple[int, int]] = []
    characters: List[str] = []
    offset = 0
    for token in sentence.tokens:
        form = "".join(token.form.split())
        start, end = offset, offset + len(form)
        offset = end
        characters.append(form)
        token_spans.append((start, end))
        for word in token.words:
            words.append(_SpanWord(word, start, end, token.is_multiword))
    return words, token_spans, "".join(characters)


def _lcs_pairs(gold: Sequence[_SpanWord], system: Sequence[_SpanWord]) -> List[Tuple[int, int]]:
    rows, cols = len(gold), len(system)
    table = [[0] * (cols + 1) for _ in range(rows + 1)]
    for g in range(rows - 1, -1, -1):
        for s in range(cols - 1, -1, -1):
            if gold[g].word.form.lower() == system[s].word.form.lower():
                table[g][s] = 1 + table[g + 1][s + 1]
            else:
                table[g][s] = max(table[g + 1][s], table[g][s + 1])
    pairs: List[Tuple[int, int]] = []
    g = s = 0
    while g < rows and s < cols:
        if gold[g].word.form.lower() == system[s].word.form.lower():
            pairs.append((g, s))
            g += 1
            s += 1
        elif table[g + 1][s] >= table[g][s + 1]:
            g += 1
        else:
            s += 1
    return pairs


def align_words(gold: Sequence[_SpanWord], system: Sequence[_SpanWord]) -> Dict[int, int]:
    """Maps gold word indexes to aligned system word indexes."""
    alignment: Dict[int, int] = {}
    gi = si = 0
    while gi < len(gold) and si < len(system):
        if gold[gi].multiword or system[si].multiword:
            end = max(
                gold[gi].end if gold[gi].multiword else 0,
                system[si].end if system[si].multiword else 0,
            )
            g0, s0 = gi, si
            changed = True
            while changed:
                changed = False
                # Multi-word tokens that start inside the span extend it; ordinary
                # words join only when they lie completely inside it.
                while gi < len(gold) and (
                    gold[gi].start < end if gold[gi].multiword else gold[gi].end <= end
                ):
                    end = max(end, gold[gi].end)
                    gi += 1
                    changed = True
                while si < len(system) and (
                    system[si].start < end if system[si].multiword else system[si].end <= end
                ):
                    end = max(end, system[si].end)
                    si += 1
                    changed = True
            for g, s in _lcs_pairs(gold[g0:gi], system[s0:si]):
                alignment[g0 + g] = s0 + s
            continue
        g_word, s_word = gold[gi], system[si]
        if (g_word.start, g_word.end) == (s_word.start, s_word.end):
            alignment[gi] = si
            gi += 1
            si += 1
            continue
        g_end, s_end = g_word.end, s_word.end
        if g_end <= s_end:
            gi += 1
        if s_end <= g_end:
            si += 1
    return alignment


def _universal_feats(feats: str) -> frozenset:
    if feats in ("", "_"):
        return frozenset()
    return frozenset(
        item for item in feats.split("|") if item.split("=", 1)[0] in UNIVERSAL_FEATURES
    )


@dataclass
class _Counts:
    gold: int = 0
    system: int = 0
    aligned: int = 0
    correct: Optional[Dict[str, int]] = None

    def __post_init__(self) -> None:
        if self.correct is None:
            self.correct = {metric: 0 for metric in METRICS}


def _score_sentence(gold: Sentence, system: Sentence, counts: _Counts) -> None:
    gold_words, gold_tokens, gold_chars = _span_words(gold)
    system_words, system_tokens, system_chars = _span_words(system)
    if gold_chars != system_chars:
        raise SyntaxAlignmentError(
            f"{gold.sent_id}: system tokens do not reproduce the gold text "
            f"({system_chars[:60]!r} vs {gold_chars[:60]!r})"
        )
    counts.gold += len(gold_words)
    counts.system += len(system_words)
    counts.correct["tokens"] += len(set(gold_tokens) & set(system_tokens))
    alignment = align_words(gold_words, system_words)
    system_to_gold = {s: g for g, s in alignment.items()}
    counts.aligned += len(alignment)
    gold_index_by_id = {span.word.id: index for index, span in enumerate(gold_words)}
    system_index_by_id = {span.word.id: index for index, span in enumerate(system_words)}
    for g, s in alignment.items():
        gold_word, system_word = gold_words[g].word, system_words[s].word
        counts.correct["words"] += 1
        counts.correct["upos"] += gold_word.upos == system_word.upos
        counts.correct["ufeats"] += _universal_feats(gold_word.feats) == _universal_feats(
            system_word.feats
        )
        counts.correct["lemma"] += gold_word.lemma == "_" or gold_word.lemma == system_word.lemma
        if gold_word.head == 0:
            head_matches = system_word.head == 0
        else:
            system_head = system_index_by_id.get(system_word.head)
            head_matches = (
                system_word.head != 0
                and system_head is not None
                and system_to_gold.get(system_head) == gold_index_by_id.get(gold_word.head)
            )
        if head_matches:
            counts.correct["uas"] += 1
            if gold_word.deprel.split(":")[0] == system_word.deprel.split(":")[0]:
                counts.correct["las"] += 1


def _f1(correct: int, gold: int, system: int) -> Dict[str, float]:
    precision = correct / system if system else (1.0 if not gold else 0.0)
    recall = correct / gold if gold else (1.0 if not system else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def score_syntax(
    gold_sentences: Sequence[Sentence], system_sentences: Sequence[Sentence]
) -> Dict[str, object]:
    """Scores system sentences against gold sentences matched by ``sent_id``.

    A gold sentence without a system sentence counts as all words missed.
    """
    system_by_id: Mapping[str, Sentence] = {sentence.sent_id: sentence for sentence in system_sentences}
    unknown = sorted(set(system_by_id) - {sentence.sent_id for sentence in gold_sentences})
    if unknown:
        raise SyntaxAlignmentError(f"system sentences not in gold: {unknown[:5]}")
    counts = _Counts()
    gold_token_total = system_token_total = 0
    missing = 0
    for gold in gold_sentences:
        gold_token_total += len(gold.tokens)
        system = system_by_id.get(gold.sent_id)
        if system is None:
            missing += 1
            counts.gold += len(gold.words)
            continue
        system_token_total += len(system.tokens)
        _score_sentence(gold, system, counts)
    metrics: Dict[str, object] = {}
    for metric in METRICS:
        if metric == "tokens":
            scores = _f1(counts.correct["tokens"], gold_token_total, system_token_total)
        else:
            scores = _f1(counts.correct[metric], counts.gold, counts.system)
            scores["alignedAccuracy"] = (
                counts.correct[metric] / counts.aligned if counts.aligned else 0.0
            )
        metrics[metric] = scores
    return {
        "sentenceCount": len(gold_sentences),
        "missingSentences": missing,
        "goldWords": counts.gold,
        "systemWords": counts.system,
        "alignedWords": counts.aligned,
        "metrics": metrics,
    }
