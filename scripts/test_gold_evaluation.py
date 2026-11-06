from __future__ import annotations

import hashlib
import io
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.backends.stanza_sidecar import sentence_from_response, system_info  # noqa: E402
from evaluation.gold import entities, retrieval  # noqa: E402
from evaluation.gold.conllu import Sentence, Token, Word, parse_conllu, read_conllu, write_conllu  # noqa: E402
from evaluation.gold.gates import check_gates, load_gates  # noqa: E402
from evaluation.gold.manifest import (  # noqa: E402
    GoldCorpusError,
    corpus_by_id,
    fetch,
    load_manifest,
    verify,
)
from evaluation.gold.scores import ScoreRecord, SystemInfo, read_report, render_table  # noqa: E402
from evaluation.gold.syntax import SyntaxAlignmentError, score_syntax  # noqa: E402

FIXTURES = ROOT / "evaluation/gold/fixtures"


def f1(result, metric):
    return result["metrics"][metric]["f1"]


class ManifestTests(unittest.TestCase):
    def test_every_corpus_is_pinned_and_licensed(self) -> None:
        corpora = load_manifest()
        self.assertEqual({c.language for c in corpora if c.task == "syntax"}, {"en", "de", "es"})
        self.assertEqual({c.language for c in corpora if c.task == "entities"}, {"en", "de", "es"})
        self.assertEqual({c.language for c in corpora if c.task == "retrieval"}, {"en", "de", "es"})
        for corpus in corpora:
            self.assertIn(corpus.source_revision, corpus.url)
            self.assertTrue(corpus.license and corpus.license_url)
            self.assertEqual(corpus.split, "test")

    def test_fixture_slices_name_a_manifest_file(self) -> None:
        files = {corpus.file for corpus in load_manifest()}
        for path in FIXTURES.iterdir():
            if path.name != "README.md":
                self.assertIn(path.name, files)

    def test_fetch_keeps_only_matching_digests(self) -> None:
        payload = b"gold\n"
        corpus = replace(
            corpus_by_id("xquad-en"), sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload)
        )
        with tempfile.TemporaryDirectory() as directory:
            gold_dir = Path(directory)
            with self.assertRaises(GoldCorpusError):
                verify(corpus, gold_dir)
            with self.assertRaises(GoldCorpusError):
                fetch(corpus, gold_dir, opener=lambda url: io.BytesIO(b"tampered\n"))
            self.assertEqual(list(corpus.path(gold_dir).parent.iterdir()), [])
            path = fetch(corpus, gold_dir, opener=lambda url: io.BytesIO(payload))
            self.assertEqual(verify(corpus, gold_dir), path)
            fetch(corpus, gold_dir, opener=lambda url: self.fail("verified file re-downloaded"))


def _sentence(sent_id: str, rows):
    """rows: (token form, [(form, upos, head, deprel)])."""
    sentence = Sentence(sent_id, None)
    next_id = 1
    for form, words in rows:
        token_words = []
        for word_form, upos, head, deprel in words:
            token_words.append(Word(next_id, word_form, word_form.lower(), upos, "_", "_", head, deprel))
            next_id += 1
        sentence.tokens.append(Token(form, token_words))
    return sentence


class SyntaxTests(unittest.TestCase):
    def test_fixtures_round_trip_and_score_perfectly(self) -> None:
        for name in ("en_ewt-ud-test.conllu", "de_gsd-ud-test.conllu", "es_ancora-ud-test.conllu"):
            gold = read_conllu(FIXTURES / name)
            self.assertTrue(any(token.is_multiword for s in gold for token in s.tokens), name)
            buffer = io.StringIO()
            write_conllu(gold, buffer)
            again = list(parse_conllu(io.StringIO(buffer.getvalue())))
            self.assertEqual([s.words for s in again], [s.words for s in gold])
            result = score_syntax(gold, again)
            for metric in ("tokens", "words", "upos", "ufeats", "lemma", "uas", "las"):
                self.assertEqual(f1(result, metric), 1.0, (name, metric))

    def test_multiword_tokens_align_by_form(self) -> None:
        gold = _sentence("s", [("zum", [("zu", "ADP", 3, "case"), ("dem", "DET", 3, "det")]), ("Haus", [("Haus", "NOUN", 0, "root")])])
        # The system expands the contraction differently and keeps only one word.
        system = _sentence("s", [("zum", [("zu", "ADP", 2, "case")]), ("Haus", [("Haus", "NOUN", 0, "root")])])
        system.tokens[0] = Token("zum", [Word(1, "zum", "zu", "ADP", "_", "_", 2, "case")])
        result = score_syntax([gold], [system])
        self.assertEqual(result["alignedWords"], 1)
        self.assertEqual(result["goldWords"], 3)
        self.assertEqual(result["systemWords"], 2)
        self.assertAlmostEqual(f1(result, "tokens"), 1.0)
        expanded = _sentence("s", [("zum", [("zu", "ADP", 3, "case"), ("dem", "DET", 3, "nmod")]), ("Haus", [("Haus", "NOUN", 0, "root")])])
        result = score_syntax([gold], [expanded])
        self.assertEqual(result["alignedWords"], 3)
        self.assertAlmostEqual(f1(result, "uas"), 1.0)
        self.assertAlmostEqual(f1(result, "las"), 2 / 3)

    def test_words_crossing_a_multiword_span_stay_unaligned(self) -> None:
        # Gold `al` (a + el) covers [0, 2); the system word `ala` covers [0, 3).
        gold = _sentence("s", [("al", [("a", "ADP", 3, "case"), ("el", "DET", 3, "det")]), ("a", [("a", "NOUN", 0, "root")])])
        system = _sentence("s", [("ala", [("a", "NOUN", 0, "root")])])
        result = score_syntax([gold], [system])
        self.assertEqual(result["alignedWords"], 0)

    def test_tokenization_errors_cost_precision_and_recall(self) -> None:
        gold = _sentence("s", [("New", [("New", "PROPN", 2, "compound")]), ("York", [("York", "PROPN", 0, "root")])])
        system = _sentence("s", [("NewYork", [("NewYork", "PROPN", 0, "root")])])
        result = score_syntax([gold], [system])
        self.assertEqual(result["alignedWords"], 0)
        self.assertEqual(f1(result, "words"), 0.0)
        with self.assertRaises(SyntaxAlignmentError):
            score_syntax([gold], [_sentence("s", [("Old", [("Old", "ADJ", 0, "root")])])])

    def test_heads_must_point_at_the_aligned_gold_head(self) -> None:
        gold = _sentence("s", [("a", [("a", "DET", 2, "det")]), ("b", [("b", "NOUN", 0, "root")]), ("c", [("c", "PUNCT", 2, "punct")])])
        system = _sentence("s", [("a", [("a", "DET", 3, "det")]), ("b", [("b", "NOUN", 0, "root")]), ("c", [("c", "PUNCT", 2, "punct:x")])])
        result = score_syntax([gold], [system])
        self.assertAlmostEqual(f1(result, "uas"), 2 / 3)
        self.assertAlmostEqual(f1(result, "las"), 2 / 3)  # subtypes are ignored

    def test_missing_sentences_count_as_missed(self) -> None:
        gold = read_conllu(FIXTURES / "en_ewt-ud-test.conllu")
        result = score_syntax(gold, gold[1:])
        self.assertEqual(result["missingSentences"], 1)
        self.assertLess(f1(result, "upos"), 1.0)
        self.assertEqual(result["metrics"]["upos"]["precision"], 1.0)


class StanzaAdapterTests(unittest.TestCase):
    def test_sidecar_tokens_become_conllu_with_code_point_offsets(self) -> None:
        text = "Grüße zum Fest."
        tokens = [
            {"start_char": 0, "end_char": 5, "text": "Grüße", "words": [{"id": 1, "text": "Grüße", "lemma": "Gruß", "upos": "NOUN", "feats": "Case=Nom", "head": 0, "deprel": "root"}]},
            {"start_char": 6, "end_char": 9, "text": "zum", "words": [
                {"id": 2, "text": "zu", "lemma": "zu", "upos": "ADP", "head": 4, "deprel": "case"},
                {"id": 3, "text": "dem", "lemma": "der", "upos": "DET", "head": 4, "deprel": "det"},
            ]},
            {"start_char": 10, "end_char": 14, "text": "Fest", "words": [{"id": 4, "text": "Fest", "lemma": "Fest", "upos": "NOUN", "head": 1, "deprel": "nmod"}]},
            {"start_char": 14, "end_char": 15, "text": ".", "words": [{"id": 5, "text": "<UNK>", "lemma": ".", "upos": "PUNCT", "head": 1, "deprel": "punct"}]},
        ]
        sentence = sentence_from_response("s1", text, tokens)
        self.assertEqual([t.form for t in sentence.tokens], ["Grüße", "zum", "Fest", "."])
        self.assertEqual([t.space_after for t in sentence.tokens], [True, True, False, True])
        self.assertEqual(sentence.tokens[3].words[0].form, ".")
        self.assertTrue(sentence.tokens[1].is_multiword)
        self.assertEqual(sentence.tokens[1].words[1].feats, "_")
        buffer = io.StringIO()
        write_conllu([sentence], buffer)
        self.assertIn("2-3\tzum", buffer.getvalue())
        self.assertIn("SpaceAfter=No", buffer.getvalue())

    def test_system_info_records_model_pins(self) -> None:
        health = {
            "stanza_version": "1.15.0",
            "protocol": 1,
            "resources_sha256": "r" * 64,
            "languages": {"de": {"ready": True, "repository": "stanfordnlp/stanza-de", "revision": "a" * 40, "packages": {"pos": "gsd"}}},
        }
        info = SystemInfo.from_json(system_info(health, "de"))
        self.assertEqual((info.model, info.revision), ("stanfordnlp/stanza-de", "a" * 40))
        self.assertEqual(info.details["stanzaVersion"], "1.15.0")
        with self.assertRaises(RuntimeError):
            system_info(health, "en")


class EntityTests(unittest.TestCase):
    def test_conll02_entities_are_byte_spans_of_the_space_joined_text(self) -> None:
        with (FIXTURES / "es-europarl.test.conll02").open(encoding="utf-8") as handle:
            sentences = list(entities.parse_conll02(handle, "europarl-ner-es-test", "es"))
        self.assertEqual(len(sentences), 3)
        for sentence in sentences:
            raw = sentence.text.encode("utf-8")
            for entity in sentence.entities:
                surface = raw[entity.byte_start : entity.byte_end].decode("utf-8")
                self.assertEqual(surface, surface.strip())
        first = sentences[0].as_dict()["entities"][0]
        self.assertEqual((first["label"], first["text"]), ("ORG", "Parlamento Europeo"))

    def test_uner_entities_use_the_original_text(self) -> None:
        with (FIXTURES / "de_pud-ud-test.iob2").open(encoding="utf-8") as handle:
            sentences = list(entities.parse_uner_iob2(handle, "uner-de-pud-test", "de"))
        self.assertTrue(sentences[0].text.startswith("„"))
        texts = [e["text"] for s in sentences for e in s.as_dict()["entities"]]
        self.assertIn("Kori Schulman", texts)

    def test_iob1_and_iob2_runs(self) -> None:
        self.assertEqual(entities._iob_entities(["I-PER", "I-PER", "O", "B-ORG", "I-ORG", "I-LOC"]), [(0, 2, "PER"), (3, 5, "ORG"), (5, 6, "LOC")])
        self.assertEqual(entities._iob_entities(["B-PER", "B-PER"]), [(0, 1, "PER"), (1, 2, "PER")])

    def test_scoring_is_exact_and_ignores_unscored_types(self) -> None:
        gold = [entities.EntitySentence("a", "en", "Ada met Bob in Rome", (
            entities.EntitySpan(0, 3, "PER"), entities.EntitySpan(8, 11, "PER"),
            entities.EntitySpan(15, 19, "LOC"), entities.EntitySpan(15, 19, "MISC"),
        ))]
        predictions = {"a": [
            entities.EntitySpan(0, 3, "PER"), entities.EntitySpan(8, 10, "PER"),
            entities.EntitySpan(15, 19, "LOC"), entities.EntitySpan(4, 7, "DATE"),
        ]}
        result = entities.score_entities(gold, predictions)
        self.assertAlmostEqual(result["metrics"]["entities"]["f1"], 2 / 3)
        self.assertEqual(result["perType"]["LOC"]["f1"], 1.0)
        with self.assertRaises(entities.EntityCorpusError):
            entities.score_entities(gold, {})


class RetrievalTests(unittest.TestCase):
    def test_parallel_paragraphs_share_keys_across_languages(self) -> None:
        passages_en, queries_en = retrieval.read_xquad(FIXTURES / "xquad.en.json", "en")
        passages_de, queries_de = retrieval.read_xquad(FIXTURES / "xquad.de.json", "de")
        self.assertEqual([p.id.split(":")[1] for p in passages_en], [p.id.split(":")[1] for p in passages_de])
        perfect = {q.id: [retrieval.passage_id("en", q.passage_key)] for q in queries_de}
        result = retrieval.score_rankings(queries_de, "en", perfect)
        self.assertEqual(result["metrics"]["mrrAt10"], 1.0)
        reversed_rankings = {q.id: [p.id for p in reversed(passages_en)] for q in queries_de}
        result = retrieval.score_rankings(queries_de, "en", reversed_rankings)
        self.assertAlmostEqual(result["metrics"]["recallAt1"], 0.5)
        self.assertAlmostEqual(result["metrics"]["mrrAt10"], 0.75)


class GateTests(unittest.TestCase):
    def _report(self, directory: Path, las: float, model: str = "stanfordnlp/stanza-de", corpus_id: str = "ud-de-gsd-test-2.17") -> None:
        corpus = corpus_by_id(corpus_id)
        record = ScoreRecord(
            "syntax", "de", [corpus], SystemInfo("stanza-sidecar", model, "a" * 40), {"metrics": {"las": {"f1": las}}}
        )
        (directory / "report.json").write_text(json.dumps(record.as_dict()), encoding="utf-8")

    def _gates(self, directory: Path) -> Path:
        path = directory / "gates.json"
        path.write_text(json.dumps({"schemaVersion": 1, "gates": [{
            "id": "syntax-stanza", "capability": "syntax", "backend": "stanza-sidecar", "status": "proposed",
            "models": {"de": {"model": "stanfordnlp/stanza-de", "revision": "a" * 40}},
            "requirements": [{"language": "de", "corpus": "ud-de-gsd-test-2.17", "metric": "las", "threshold": 0.8, "report": "report.json"}],
        }]}), encoding="utf-8")
        return path

    def test_gate_passes_only_for_the_gated_model_and_pinned_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            gates = load_gates(self._gates(directory))
            self.assertFalse(check_gates(gates, directory)[0].passed)  # no report yet
            self._report(directory, 0.85)
            self.assertTrue(check_gates(gates, directory)[0].passed)
            self._report(directory, 0.75)
            self.assertIn("< 0.8", check_gates(gates, directory)[0].problem)
            self._report(directory, float("nan"))
            self.assertFalse(check_gates(gates, directory)[0].passed)
            self._report(directory, float("inf"))
            self.assertFalse(check_gates(gates, directory)[0].passed)
            self._report(directory, 0.9, model="other/model")
            self.assertIn("not the gated model", check_gates(gates, directory)[0].problem)
            self._report(directory, 0.9)
            stale = [replace(c, sha256="0" * 64) if c.id == "ud-de-gsd-test-2.17" else c for c in load_manifest()]
            self.assertIn("no longer pinned", check_gates(gates, directory, stale)[0].problem)

    def test_committed_gates_parse(self) -> None:
        load_gates()

    def test_table_lists_each_report(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self._report(directory, 0.9)
            table = render_table([read_report(directory / "report.json")])
        self.assertIn("| stanza-sidecar | stanfordnlp/stanza-de @ aaaaaaaa | de | ud-de-gsd-test-2.17 |", table)


if __name__ == "__main__":
    unittest.main()
