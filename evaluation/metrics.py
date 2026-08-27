from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import log2, sqrt
from typing import Dict, Hashable, Iterable, List, Sequence, TypeVar, Union

T = TypeVar("T", bound=Hashable)


@dataclass(frozen=True)
class PrecisionRecallF1:
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int

    def as_dict(self) -> Dict[str, Union[float, int]]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "truePositives": self.true_positives,
            "falsePositives": self.false_positives,
            "falseNegatives": self.false_negatives,
        }


def precision_recall_f1(gold: Iterable[T], predicted: Iterable[T]) -> PrecisionRecallF1:
    gold_counts = Counter(gold)
    predicted_counts = Counter(predicted)
    true_positives = sum((gold_counts & predicted_counts).values())
    false_positives = sum((predicted_counts - gold_counts).values())
    false_negatives = sum((gold_counts - predicted_counts).values())

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = (
        true_positives / precision_denominator
        if precision_denominator
        else (1.0 if not gold_counts else 0.0)
    )
    recall = (
        true_positives / recall_denominator
        if recall_denominator
        else (1.0 if not gold_counts else 0.0)
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return PrecisionRecallF1(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def accuracy(gold: Sequence[T], predicted: Sequence[T]) -> float:
    if len(gold) != len(predicted):
        raise ValueError("accuracy requires equal-length sequences")
    if not gold:
        return 1.0
    return sum(left == right for left, right in zip(gold, predicted)) / len(gold)


def macro_f1(gold: Sequence[T], predicted: Sequence[T]) -> float:
    if len(gold) != len(predicted):
        raise ValueError("macro_f1 requires equal-length sequences")
    labels = set(gold) | set(predicted)
    if not labels:
        return 1.0
    scores = []
    for label in labels:
        gold_binary = [index for index, value in enumerate(gold) if value == label]
        predicted_binary = [index for index, value in enumerate(predicted) if value == label]
        scores.append(precision_recall_f1(gold_binary, predicted_binary).f1)
    return sum(scores) / len(scores)


def reciprocal_rank(ranking: Sequence[T], relevant: Iterable[T]) -> float:
    relevant_set = set(relevant)
    for index, item in enumerate(ranking, start=1):
        if item in relevant_set:
            return 1.0 / index
    return 0.0


def mean_reciprocal_rank(
    rankings: Sequence[Sequence[T]],
    relevant: Sequence[Iterable[T]],
) -> float:
    if len(rankings) != len(relevant):
        raise ValueError("mean_reciprocal_rank requires one relevance set per ranking")
    if not rankings:
        return 1.0
    return sum(
        reciprocal_rank(ranking, relevant_items)
        for ranking, relevant_items in zip(rankings, relevant)
    ) / len(rankings)


def recall_at_k(ranking: Sequence[T], relevant: Iterable[T], k: int) -> float:
    if k < 0:
        raise ValueError("k must be non-negative")
    relevant_set = set(relevant)
    if not relevant_set:
        return 1.0
    found = relevant_set & set(ranking[:k])
    return len(found) / len(relevant_set)


def ndcg_at_k(ranking: Sequence[T], relevant: Iterable[T], k: int) -> float:
    if k < 0:
        raise ValueError("k must be non-negative")
    relevant_set = set(relevant)
    if not relevant_set:
        return 1.0

    dcg = sum(
        1.0 / log2(index + 2)
        for index, item in enumerate(ranking[:k])
        if item in relevant_set
    )
    ideal_hits = min(len(relevant_set), k)
    if ideal_hits == 0:
        return 0.0
    ideal = sum(1.0 / log2(index + 2) for index in range(ideal_hits))
    return dcg / ideal


def _average_ranks(values: Sequence[float]) -> List[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average = ((cursor + 1) + end) / 2.0
        for original_index, _ in indexed[cursor:end]:
            ranks[original_index] = average
        cursor = end
    return ranks


def spearman_correlation(gold: Sequence[float], predicted: Sequence[float]) -> float:
    if len(gold) != len(predicted):
        raise ValueError("spearman_correlation requires equal-length sequences")
    if not gold:
        return 1.0

    gold_ranks = _average_ranks(gold)
    predicted_ranks = _average_ranks(predicted)
    gold_mean = sum(gold_ranks) / len(gold_ranks)
    predicted_mean = sum(predicted_ranks) / len(predicted_ranks)

    numerator = sum(
        (gold_rank - gold_mean) * (predicted_rank - predicted_mean)
        for gold_rank, predicted_rank in zip(gold_ranks, predicted_ranks)
    )
    gold_variance = sum((rank - gold_mean) ** 2 for rank in gold_ranks)
    predicted_variance = sum((rank - predicted_mean) ** 2 for rank in predicted_ranks)
    denominator = sqrt(gold_variance * predicted_variance)
    if denominator == 0.0:
        return 1.0 if gold_ranks == predicted_ranks else 0.0
    return numerator / denominator
