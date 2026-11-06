"""Named-entity gold readers and entity-level scoring.

Gold sentences become ``{id, language, text, entities}`` records whose entities are
UTF-8 half-open byte spans, the coordinate system nlp-stack uses everywhere.
Scoring is exact span and type match (CoNLL entity-level micro F1) over the
scored types only; gold ``MISC`` and system labels outside the scored set are
dropped on both sides, so models without a MISC class are comparable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from evaluation.metrics import precision_recall_f1

SCORED_TYPES = ("PER", "ORG", "LOC")


class EntityCorpusError(ValueError):
    pass


@dataclass(frozen=True)
class EntitySpan:
    byte_start: int
    byte_end: int
    label: str

    def as_dict(self, text: Optional[str] = None) -> Dict[str, object]:
        value: Dict[str, object] = {
            "byteStart": self.byte_start,
            "byteEnd": self.byte_end,
            "label": self.label,
        }
        if text is not None:
            value["text"] = text.encode("utf-8")[self.byte_start : self.byte_end].decode("utf-8")
        return value


@dataclass(frozen=True)
class EntitySentence:
    id: str
    language: str
    text: str
    entities: Tuple[EntitySpan, ...]

    def as_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "language": self.language,
            "text": self.text,
            "entities": [entity.as_dict(self.text) for entity in self.entities],
        }


def _iob_entities(tags: Sequence[str]) -> List[Tuple[int, int, str]]:
    """Token-index entities from IOB1 or IOB2 tags (an ``I-`` after O or another type starts one)."""
    entities: List[Tuple[int, int, str]] = []
    start: Optional[int] = None
    label: Optional[str] = None
    for index, tag in enumerate(list(tags) + ["O"]):
        prefix, _, kind = tag.partition("-")
        continues = prefix == "I" and label == kind
        if start is not None and not continues:
            entities.append((start, index, label))
            start, label = None, None
        if prefix in ("B", "I") and not continues:
            start, label = index, kind
        elif prefix not in ("B", "I", "O"):
            raise EntityCorpusError(f"unknown IOB tag {tag!r}")
    return entities


def _token_byte_spans(text: str, tokens: Sequence[str], case_id: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        start = text.find(token, cursor)
        if start < 0 or text[cursor:start].strip():
            raise EntityCorpusError(f"{case_id}: token {token!r} not found in text at {cursor}")
        end = start + len(token)
        spans.append((len(text[:start].encode("utf-8")), len(text[:end].encode("utf-8"))))
        cursor = end
    return spans


def _sentence(
    case_id: str, language: str, text: str, tokens: Sequence[str], tags: Sequence[str]
) -> EntitySentence:
    spans = _token_byte_spans(text, tokens, case_id)
    entities = tuple(
        EntitySpan(spans[start][0], spans[end - 1][1], label)
        for start, end, label in _iob_entities(tags)
    )
    return EntitySentence(case_id, language, text, entities)


def parse_conll02(lines: Iterable[str], corpus_id: str, language: str) -> Iterator[EntitySentence]:
    """CoNLL-2002 style ``token<TAB or space>tag`` lines; text is the tokens joined by spaces."""
    tokens: List[str] = []
    tags: List[str] = []
    count = 0
    for raw in list(lines) + [""]:
        line = raw.rstrip("\n").rstrip("\r")
        if not line.strip():
            if tokens:
                count += 1
                yield _sentence(f"{corpus_id}-{count:04d}", language, " ".join(tokens), tokens, tags)
            tokens, tags = [], []
            continue
        if line.startswith("-DOCSTART-"):
            continue
        parts = line.split("\t") if "\t" in line else line.split(" ")
        tokens.append(parts[0])
        tags.append(parts[-1])


def parse_uner_iob2(lines: Iterable[str], corpus_id: str, language: str) -> Iterator[EntitySentence]:
    """Universal NER files: UD ``# sent_id``/``# text`` comments, then ``id token tag`` rows."""
    sent_id: Optional[str] = None
    text: Optional[str] = None
    tokens: List[str] = []
    tags: List[str] = []
    for raw in list(lines) + [""]:
        line = raw.rstrip("\n").rstrip("\r")
        if not line.strip():
            if tokens:
                if text is None or sent_id is None:
                    raise EntityCorpusError(f"{corpus_id}: sentence without # sent_id or # text")
                yield _sentence(f"{corpus_id}-{sent_id}", language, text, tokens, tags)
            sent_id, text, tokens, tags = None, None, [], []
            continue
        if line.startswith("#"):
            key, _, value = line[1:].partition("=")
            if key.strip() == "sent_id":
                sent_id = value.strip()
            elif key.strip() == "text":
                text = value.strip()
            continue
        columns = line.split("\t")
        tokens.append(columns[1])
        tags.append(columns[2])


def score_entities(
    gold: Sequence[EntitySentence],
    predictions: Mapping[str, Sequence[EntitySpan]],
    scored_types: Sequence[str] = SCORED_TYPES,
) -> Dict[str, object]:
    """Exact span+type micro F1, overall and per type. Every gold id needs a prediction."""
    missing = sorted({sentence.id for sentence in gold} - set(predictions))
    if missing:
        raise EntityCorpusError(f"predictions missing for {len(missing)} sentences, e.g. {missing[:3]}")
    extra = sorted(set(predictions) - {sentence.id for sentence in gold})
    if extra:
        raise EntityCorpusError(f"predictions for unknown sentences, e.g. {extra[:3]}")
    allowed = set(scored_types)
    gold_events = [
        (sentence.id, entity.byte_start, entity.byte_end, entity.label)
        for sentence in gold
        for entity in sentence.entities
        if entity.label in allowed
    ]
    predicted_events = [
        (case_id, entity.byte_start, entity.byte_end, entity.label)
        for case_id, entities in predictions.items()
        for entity in entities
        if entity.label in allowed
    ]
    per_type = {
        label: precision_recall_f1(
            [event for event in gold_events if event[3] == label],
            [event for event in predicted_events if event[3] == label],
        ).as_dict()
        for label in scored_types
    }
    return {
        "sentenceCount": len(gold),
        "scoredTypes": list(scored_types),
        "metrics": {"entities": precision_recall_f1(gold_events, predicted_events).as_dict()},
        "perType": per_type,
    }


def entity_spans_from_json(values: Sequence[Mapping[str, object]]) -> List[EntitySpan]:
    spans: List[EntitySpan] = []
    for value in values:
        start, end, label = value.get("byteStart"), value.get("byteEnd"), value.get("label")
        if not isinstance(start, int) or not isinstance(end, int) or not isinstance(label, str):
            raise EntityCorpusError(f"bad entity prediction {value!r}")
        spans.append(EntitySpan(start, end, label))
    return spans
