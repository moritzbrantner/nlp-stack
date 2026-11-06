"""Gold-set command line: fetch, export adapter inputs, score predictions, print tables, check gates.

    python3 -m evaluation.gold fetch [CORPUS ...]
    python3 -m evaluation.gold export --out DIR
    python3 -m evaluation.gold score-syntax --corpus ID --predictions FILE.conllu --systems FILE
    python3 -m evaluation.gold score-entities --corpus ID --predictions FILE.jsonl --systems FILE
    python3 -m evaluation.gold score-retrieval --queries ID --passages ID --predictions FILE.jsonl --systems FILE
    python3 -m evaluation.gold table REPORT ...
    python3 -m evaluation.gold gates [--reports-root DIR]

Corpora live in ``NLP_GOLD_DIR`` (default ``.test-corpora/gold``, ignored by git).
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from evaluation.gold import entities as entity_gold
from evaluation.gold import retrieval as retrieval_gold
from evaluation.gold.conllu import read_conllu
from evaluation.gold.gates import GATES_PATH, check_gates, load_gates
from evaluation.gold.manifest import (
    GoldCorpus,
    GoldCorpusError,
    REPOSITORY_ROOT,
    corpus_by_id,
    fetch,
    gold_dir_from_env,
    load_manifest,
    verify,
)
from evaluation.gold.scores import ScoreRecord, load_systems, read_report, render_table
from evaluation.gold.syntax import score_syntax


def read_entity_corpus(corpus: GoldCorpus, gold_dir: Path) -> List[entity_gold.EntitySentence]:
    path = verify(corpus, gold_dir)
    with path.open(encoding="utf-8") as handle:
        if corpus.format == "conll02":
            return list(entity_gold.parse_conll02(handle, corpus.id, corpus.language))
        if corpus.format == "uner-iob2":
            return list(entity_gold.parse_uner_iob2(handle, corpus.id, corpus.language))
    raise GoldCorpusError(f"{corpus.id}: not an entity corpus")


def _write_jsonl(path: Path, records) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def export_inputs(corpora: Sequence[GoldCorpus], gold_dir: Path, out: Path) -> None:
    """Writes the raw-text inputs adapters need; gold labels stay out of these files."""
    for corpus in corpora:
        if corpus.task == "syntax":
            sentences = read_conllu(verify(corpus, gold_dir))
            records = (
                {"id": sentence.sent_id, "language": corpus.language, "text": sentence.text}
                for sentence in sentences
            )
        elif corpus.task == "entities":
            records = (
                {"id": sentence.id, "language": sentence.language, "text": sentence.text}
                for sentence in read_entity_corpus(corpus, gold_dir)
            )
        else:
            passages, queries = retrieval_gold.read_xquad(verify(corpus, gold_dir), corpus.language)
            _write_jsonl(
                out / f"{corpus.id}.passages.jsonl",
                ({"id": p.id, "language": p.language, "text": p.text} for p in passages),
            )
            records = ({"id": q.id, "language": q.language, "text": q.text} for q in queries)
            count = _write_jsonl(out / f"{corpus.id}.queries.jsonl", records)
            print(f"{corpus.id}: {len(passages)} passages, {count} queries")
            continue
        count = _write_jsonl(out / f"{corpus.id}.inputs.jsonl", records)
        print(f"{corpus.id}: {count} inputs")


def _emit(record: ScoreRecord, output: Optional[Path]) -> None:
    rendered = json.dumps(record.as_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if output is None:
        sys.stdout.write(rendered)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")


def _system_for(systems_path: Path, language: str):
    systems = load_systems(systems_path)
    if language not in systems:
        raise GoldCorpusError(f"{systems_path}: no system recorded for {language}")
    return systems[language]


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--systems", type=Path, required=True, help="{language: system} JSON")
    parser.add_argument("--source-revision")
    parser.add_argument("--output", type=Path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m evaluation.gold", description=__doc__.split("\n")[0])
    parser.add_argument("--gold-dir", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)

    fetch_parser = commands.add_parser("fetch", help="download and verify pinned corpora")
    fetch_parser.add_argument("corpora", nargs="*")
    verify_parser = commands.add_parser("verify", help="verify fetched corpora")
    verify_parser.add_argument("corpora", nargs="*")
    export_parser = commands.add_parser("export", help="write adapter inputs")
    export_parser.add_argument("--out", type=Path, required=True)
    export_parser.add_argument("corpora", nargs="*")

    syntax_parser = commands.add_parser("score-syntax")
    syntax_parser.add_argument("--corpus", required=True)
    _common(syntax_parser)
    entities_parser = commands.add_parser("score-entities")
    entities_parser.add_argument("--corpus", required=True)
    _common(entities_parser)
    retrieval_parser = commands.add_parser("score-retrieval")
    retrieval_parser.add_argument("--queries", required=True)
    retrieval_parser.add_argument("--passages", required=True)
    _common(retrieval_parser)

    table_parser = commands.add_parser("table", help="per-language Markdown table of reports")
    table_parser.add_argument("reports", nargs="+", type=Path)
    gates_parser = commands.add_parser("gates", help="check model-default gates")
    gates_parser.add_argument("--gates", type=Path, default=GATES_PATH)
    gates_parser.add_argument("--reports-root", type=Path, default=REPOSITORY_ROOT)

    args = parser.parse_args(argv)
    gold_dir = args.gold_dir or gold_dir_from_env()
    manifest = load_manifest()

    def selected(ids: Sequence[str]) -> List[GoldCorpus]:
        return [corpus_by_id(item, manifest) for item in ids] if ids else manifest

    try:
        if args.command == "fetch":
            for corpus in selected(args.corpora):
                print(f"{corpus.id}: {fetch(corpus, gold_dir)}")
            return 0
        if args.command == "verify":
            for corpus in selected(args.corpora):
                print(f"{corpus.id}: ok {verify(corpus, gold_dir)}")
            return 0
        if args.command == "export":
            export_inputs(selected(args.corpora), gold_dir, args.out)
            return 0
        measured_at = datetime.date.today().isoformat()
        if args.command == "score-syntax":
            corpus = corpus_by_id(args.corpus, manifest)
            result = score_syntax(read_conllu(verify(corpus, gold_dir)), read_conllu(args.predictions))
            record = ScoreRecord("syntax", corpus.language, [corpus], _system_for(args.systems, corpus.language), result, args.source_revision, measured_at)
            _emit(record, args.output)
            return 0
        if args.command == "score-entities":
            corpus = corpus_by_id(args.corpus, manifest)
            gold = read_entity_corpus(corpus, gold_dir)
            predictions = {}
            with args.predictions.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        value = json.loads(line)
                        predictions[value["id"]] = entity_gold.entity_spans_from_json(value["entities"])
            result = entity_gold.score_entities(gold, predictions)
            record = ScoreRecord("entities", corpus.language, [corpus], _system_for(args.systems, corpus.language), result, args.source_revision, measured_at)
            _emit(record, args.output)
            return 0
        if args.command == "score-retrieval":
            queries_corpus = corpus_by_id(args.queries, manifest)
            passages_corpus = corpus_by_id(args.passages, manifest)
            _, queries = retrieval_gold.read_xquad(verify(queries_corpus, gold_dir), queries_corpus.language)
            verify(passages_corpus, gold_dir)
            rankings = {}
            with args.predictions.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        value = json.loads(line)
                        rankings[value["id"]] = value["ranking"]
            result = retrieval_gold.score_rankings(queries, passages_corpus.language, rankings)
            record = ScoreRecord("retrieval", queries_corpus.language, [queries_corpus, passages_corpus], _system_for(args.systems, queries_corpus.language), result, args.source_revision, measured_at)
            _emit(record, args.output)
            return 0
        if args.command == "table":
            print(render_table([read_report(path) for path in args.reports]))
            return 0
        if args.command == "gates":
            failed = 0
            for result in check_gates(load_gates(args.gates), args.reports_root, manifest):
                requirement = result.requirement
                score = "-" if result.score is None else f"{result.score:.4f}"
                status = "pass" if result.passed else f"FAIL ({result.problem})"
                print(f"{result.gate} {requirement.language} {requirement.corpus} {requirement.metric} >= {requirement.threshold}: {score} {status}")
                failed += not result.passed
            return 1 if failed else 0
    except (GoldCorpusError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
