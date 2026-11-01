#!/usr/bin/env python3
"""Build the UD-derived lemma tables used by `text_linguistics::UdLemmatizer`.

The lemmatizer predicts a lemma from (surface form, UPOS). It learns suffix edit
rules ("strip k characters, append s") per UPOS from a UD training split and keeps
an exception list for the training words the rules get wrong. Only the training
split is read, so the test split stays a fair evaluation set.

Usage:
    python3 scripts/build_ud_lemma_tables.py de de_gsd-ud-train.conllu \
        --source "UD_German-GSD r2.17" --license "CC BY-SA 4.0" \
        > crates/text/text-linguistics/data/lemmas/de.tsv

Output format (tab-separated, `#` comments):
    C  UPOS                         lemmas of this UPOS start upper-case
    K  UPOS                         lemma is the surface form unchanged
    R  UPOS  suffix  strip  append  suffix edit rule (empty suffix = UPOS default)
    E  form  UPOS  lemma            exception, keyed by the lower-cased form

The rule application here must match `crates/text/text-linguistics/src/lemmas.rs`.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict

MAX_SUFFIX = 7
MIN_RULE_COUNT = 2
KEEP_FORM_UPOS = {"PROPN", "PUNCT", "NUM", "SYM", "X"}


def read_words(path: str):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if "-" in cols[0] or "." in cols[0]:
                continue
            form, lemma, upos = cols[1], cols[2], cols[3]
            if lemma == "_" and form != "_":
                continue
            yield form, upos, lemma


def edit_of(form: str, lemma: str) -> tuple[int, str]:
    prefix = 0
    for a, b in zip(form, lemma):
        if a != b:
            break
        prefix += 1
    return len(form) - prefix, lemma[prefix:]


def apply_edit(form_lower: str, edit: tuple[int, str]) -> str | None:
    strip, append = edit
    if strip > len(form_lower):
        return None
    return form_lower[: len(form_lower) - strip] + append


def recase(lemma: str, upos: str, capitalized: set[str]) -> str:
    if upos in capitalized and lemma:
        return lemma[0].upper() + lemma[1:]
    return lemma


def predict(form: str, upos: str, rules, capitalized, keep_form) -> str:
    if upos in keep_form:
        return form
    lower = form.lower()
    for length in range(min(MAX_SUFFIX, len(lower)), -1, -1):
        edit = rules.get((upos, lower[len(lower) - length :] if length else ""))
        if edit is None:
            continue
        candidate = apply_edit(lower, edit)
        if candidate:
            return recase(candidate, upos, capitalized)
    return recase(lower, upos, capitalized)


def build(words):
    lemma_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    surface_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    upos_capital: dict[str, Counter] = defaultdict(Counter)
    upos_identity: dict[str, Counter] = defaultdict(Counter)
    for form, upos, lemma in words:
        lemma_counts[(form.lower(), upos)][lemma] += 1
        surface_counts[(form.lower(), upos)][form] += 1
        upos_capital[upos][bool(lemma[:1].isupper())] += 1
        upos_identity[upos][lemma == form] += 1

    capitalized = {u for u, c in upos_capital.items() if c[True] > c[False]}
    keep_form = {
        u for u, c in upos_identity.items() if u in KEEP_FORM_UPOS and c[True] > c[False]
    }

    suffix_edits: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for (lower, upos), lemmas in lemma_counts.items():
        lemma = lemmas.most_common(1)[0][0]
        edit = edit_of(lower, lemma.lower())
        weight = sum(lemmas.values())
        for length in range(0, min(MAX_SUFFIX, len(lower)) + 1):
            suffix = lower[len(lower) - length :] if length else ""
            suffix_edits[(upos, suffix)][edit] += weight

    # Keep a rule only where it differs from the next shorter suffix's rule.
    rules: dict[tuple[str, str], tuple[int, str]] = {}
    for length in range(0, MAX_SUFFIX + 1):
        for (upos, suffix), edits in suffix_edits.items():
            if len(suffix) != length:
                continue
            edit, count = edits.most_common(1)[0]
            if length and count < MIN_RULE_COUNT:
                continue
            parent = None
            for shorter in range(length - 1, -1, -1):
                parent = rules.get((upos, suffix[len(suffix) - shorter :] if shorter else ""))
                if parent is not None:
                    break
            if length and parent == edit:
                continue
            rules[(upos, suffix)] = edit

    exceptions = {}
    for (lower, upos), lemmas in lemma_counts.items():
        lemma = lemmas.most_common(1)[0][0]
        surface = surface_counts[(lower, upos)].most_common(1)[0][0]
        if predict(surface, upos, rules, capitalized, keep_form) != lemma:
            exceptions[(lower, upos)] = lemma
    return capitalized, keep_form, rules, exceptions


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("language")
    parser.add_argument("train_conllu")
    parser.add_argument("--source", required=True)
    parser.add_argument("--license", required=True)
    args = parser.parse_args(argv)

    capitalized, keep_form, rules, exceptions = build(read_words(args.train_conllu))
    out = sys.stdout
    out.write(f"# UD lemma table for `{args.language}`, built by scripts/build_ud_lemma_tables.py\n")
    out.write(f"# Derived from the training split of {args.source} ({args.license}).\n")
    for upos in sorted(capitalized):
        out.write(f"C\t{upos}\n")
    for upos in sorted(keep_form):
        out.write(f"K\t{upos}\n")
    for (upos, suffix), (strip, append) in sorted(rules.items()):
        out.write(f"R\t{upos}\t{suffix}\t{strip}\t{append}\n")
    for (lower, upos), lemma in sorted(exceptions.items()):
        if "\t" in lower or "\t" in lemma:
            continue
        out.write(f"E\t{lower}\t{upos}\t{lemma}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
