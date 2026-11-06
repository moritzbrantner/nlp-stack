# Gold-set fixture slices

Tiny, unmodified excerpts of the pinned corpora in `../manifest.json`, used only by
the offline unit tests (`scripts/test_gold_evaluation.py`). Real evaluations fetch
the full pinned files with `python3 -m evaluation.gold fetch`.

| File | Excerpt of | Licence |
| --- | --- | --- |
| `en_ewt-ud-test.conllu` | 3 sentences of UD_English-EWT r2.17 test | CC BY-SA 4.0 |
| `de_gsd-ud-test.conllu` | 3 sentences of UD_German-GSD r2.17 test | CC BY-SA 4.0 |
| `es_ancora-ud-test.conllu` | 3 sentences of UD_Spanish-AnCora r2.17 test | CC BY 4.0 |
| `es-europarl.test.conll02` | 3 sentences of the ixa-ehu Europarl NER test set (Spanish) | Europarl terms (no known copyright restrictions) |
| `de_pud-ud-test.iob2` | 3 sentences of UNER_German-PUD | CC BY-SA 3.0 |
| `xquad.en.json`, `xquad.de.json` | first article, 2 paragraphs, 2 questions each, of XQuAD | CC BY-SA 4.0 |

Sources and pinned revisions are the manifest entries with the same file names.
