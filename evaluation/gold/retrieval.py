"""Passage retrieval gold built from XQuAD's parallel en/de/es question/paragraph pairs.

Every XQuAD question belongs to exactly one paragraph, and paragraphs are parallel
across languages (same article and paragraph position). A query in language A is
scored against the paragraphs of language B, which yields monolingual (A = B) and
cross-lingual (A != B) retrieval over the same relevance judgements.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

from evaluation.metrics import ndcg_at_k, recall_at_k, reciprocal_rank

CUTOFF = 10


class RetrievalCorpusError(ValueError):
    pass


@dataclass(frozen=True)
class Passage:
    id: str
    language: str
    text: str


@dataclass(frozen=True)
class Query:
    id: str
    language: str
    text: str
    passage_key: str


def passage_id(language: str, key: str) -> str:
    return f"{language}:{key}"


def read_xquad(path: Path, language: str) -> Tuple[List[Passage], List[Query]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    passages: List[Passage] = []
    queries: List[Query] = []
    for article_index, article in enumerate(data["data"]):
        for paragraph_index, paragraph in enumerate(article["paragraphs"]):
            key = f"{article_index:02d}-{paragraph_index:02d}"
            passages.append(Passage(passage_id(language, key), language, paragraph["context"]))
            for qa in paragraph["qas"]:
                queries.append(Query(f"{language}:{qa['id']}", language, qa["question"], key))
    return passages, queries


def score_rankings(
    queries: Sequence[Query],
    passage_language: str,
    rankings: Mapping[str, Sequence[str]],
    cutoff: int = CUTOFF,
) -> Dict[str, object]:
    """MRR@k, Recall@1, Recall@k and nDCG@k of query rankings over ``passage_language``."""
    missing = [query.id for query in queries if query.id not in rankings]
    if missing:
        raise RetrievalCorpusError(f"rankings missing for {len(missing)} queries, e.g. {missing[:3]}")
    reciprocal = recall1 = recallk = ndcg = 0.0
    for query in queries:
        ranking = list(rankings[query.id])[:cutoff]
        relevant = [passage_id(passage_language, query.passage_key)]
        reciprocal += reciprocal_rank(ranking, relevant)
        recall1 += recall_at_k(ranking, relevant, 1)
        recallk += recall_at_k(ranking, relevant, cutoff)
        ndcg += ndcg_at_k(ranking, relevant, cutoff)
    count = len(queries) or 1
    return {
        "queryCount": len(queries),
        "queryLanguage": queries[0].language if queries else None,
        "passageLanguage": passage_language,
        "cutoff": cutoff,
        "metrics": {
            f"mrrAt{cutoff}": reciprocal / count,
            "recallAt1": recall1 / count,
            f"recallAt{cutoff}": recallk / count,
            f"ndcgAt{cutoff}": ndcg / count,
        },
    }
