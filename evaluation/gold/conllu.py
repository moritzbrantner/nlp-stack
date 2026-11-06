"""Minimal typed CoNLL-U reader and writer for UD evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, TextIO


@dataclass(frozen=True)
class Word:
    """A syntactic word (integer CoNLL-U ID)."""

    id: int
    form: str
    lemma: str
    upos: str
    xpos: str
    feats: str
    head: int
    deprel: str


@dataclass(frozen=True)
class Token:
    """A surface token: one word, or a multi-word token covering several words."""

    form: str
    words: List[Word]
    space_after: bool = True

    @property
    def is_multiword(self) -> bool:
        return len(self.words) > 1


@dataclass
class Sentence:
    sent_id: str
    text: Optional[str]
    tokens: List[Token] = field(default_factory=list)

    @property
    def words(self) -> List[Word]:
        return [word for token in self.tokens for word in token.words]


class ConlluError(ValueError):
    pass


def _space_after(misc: str) -> bool:
    return "SpaceAfter=No" not in misc.split("|")


def parse_conllu(lines: Iterable[str], source: str = "<conllu>") -> Iterator[Sentence]:
    sent_id: Optional[str] = None
    text: Optional[str] = None
    rows: List[List[str]] = []
    count = 0

    def finish() -> Sentence:
        nonlocal count
        count += 1
        sentence = Sentence(sent_id or f"{source}#{count}", text)
        pending_range: Optional[tuple] = None
        pending_words: List[Word] = []
        for row in rows:
            if "-" in row[0]:
                start, end = (int(part) for part in row[0].split("-"))
                pending_range = (start, end, row[1], _space_after(row[9]))
                pending_words = []
                continue
            word = Word(
                id=int(row[0]),
                form=row[1],
                lemma=row[2],
                upos=row[3],
                xpos=row[4],
                feats=row[5],
                head=int(row[6]) if row[6] != "_" else 0,
                deprel=row[7],
            )
            if pending_range is not None:
                pending_words.append(word)
                if word.id == pending_range[1]:
                    sentence.tokens.append(Token(pending_range[2], pending_words, pending_range[3]))
                    pending_range = None
                continue
            sentence.tokens.append(Token(word.form, [word], _space_after(row[9])))
        if pending_range is not None:
            raise ConlluError(f"{source}: {sentence.sent_id}: unterminated multi-word token")
        return sentence

    for number, raw in enumerate(lines, start=1):
        line = raw.rstrip("\n").rstrip("\r")
        if not line:
            if rows:
                yield finish()
            sent_id, text, rows = None, None, []
            continue
        if line.startswith("#"):
            key, _, value = line[1:].partition("=")
            if key.strip() == "sent_id":
                sent_id = value.strip()
            elif key.strip() == "text":
                text = value.strip()
            continue
        columns = line.split("\t")
        if len(columns) != 10:
            raise ConlluError(f"{source}:{number}: expected 10 columns")
        if "." in columns[0]:
            continue  # empty nodes are not scored
        rows.append(columns)
    if rows:
        yield finish()


def read_conllu(path: Path) -> List[Sentence]:
    with path.open(encoding="utf-8") as handle:
        return list(parse_conllu(handle, source=str(path)))


def write_conllu(sentences: Iterable[Sentence], output: TextIO) -> None:
    for sentence in sentences:
        output.write(f"# sent_id = {sentence.sent_id}\n")
        if sentence.text is not None:
            output.write(f"# text = {sentence.text}\n")
        for token in sentence.tokens:
            misc = "_" if token.space_after else "SpaceAfter=No"
            if token.is_multiword:
                first, last = token.words[0].id, token.words[-1].id
                output.write(f"{first}-{last}\t{token.form}\t_\t_\t_\t_\t_\t_\t_\t{misc}\n")
            for word in token.words:
                word_misc = misc if not token.is_multiword else "_"
                output.write(
                    "\t".join(
                        [
                            str(word.id),
                            word.form,
                            word.lemma or "_",
                            word.upos or "_",
                            word.xpos or "_",
                            word.feats or "_",
                            str(word.head),
                            word.deprel or "_",
                            "_",
                            word_misc,
                        ]
                    )
                    + "\n"
                )
        output.write("\n")
