"""Text folding for matching names across case, diacritics and Turkish dotted/dotless i.

FalkorDB's full-text index does not case-fold non-ASCII letters ("şimşek" does
not find "Şimşek"), so names and chunks are indexed in folded form.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_HYPHENATED_KEY = re.compile(r"\b([A-Za-z]+(?:-[0-9]+)+)\b(?:['’‘][^\W\d_]+)?")

DEFAULT_NAME_SUFFIXES = (
    "a s",
    "as",
    "anonim sirketi",
    "ltd sti",
    "ltd",
    "sti",
    "limited sirketi",
    "limited",
    "koll sti",
    "kom sti",
    "inc",
    "incorporated",
    "llc",
    "plc",
    "corp",
    "corporation",
    "co",
    "ltd co",
    "gmbh",
    "ag",
    "kg",
    "sa",
    "s a",
    "bv",
    "nv",
    "srl",
    "spa",
)

# Too common in questions to say anything about which entity is meant.
STOPWORDS = frozenset(
    (
        # Turkish
        "ve ile veya ama icin gibi kadar daha cok en bir bu su o ne nedir neler nasil "
        "hangi hangisi kim kimdir kimin nerede zaman mi mu midir mudur olan olarak var "
        "yok ise de da ki "
        # English
        "the and for with what which who whom how when where does from into that this "
        "are was were has have its their there"
    ).split()
)


def fold(text: str) -> str:
    """Lowercase ASCII skeleton: 'İnşaat A.Ş.' -> 'insaat a s'."""
    decomposed = unicodedata.normalize("NFKD", text.replace("ı", "i").replace("İ", "I"))
    ascii_only = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return _NON_ALNUM.sub(" ", ascii_only.casefold()).strip()


def match_name(text: str, suffixes: Sequence[str]) -> str:
    """Fold a name and strip folded suffixes while a base name remains."""
    name = fold(text)
    endings = sorted((f" {suffix}" for suffix in suffixes if suffix), key=len, reverse=True)
    while True:
        for ending in endings:
            if name.endswith(ending):
                name = name.removesuffix(ending)
                break
        else:
            return name


def fulltext_query(question: str, *, prefix_len: int = 6) -> str:
    """OR-query of folded prefixes and exact phrases for hyphenated keys."""
    terms = [f'"{fold(key)}"' for key in _HYPHENATED_KEY.findall(question)]
    for word in fold(_HYPHENATED_KEY.sub(" ", question)).split():
        if len(word) < 3 or word in STOPWORDS:
            continue
        terms.append(word[:prefix_len] + "*")
    return "|".join(dict.fromkeys(terms))


def bm25_tokens(text: str) -> list[str]:
    """Tokenize chunks and questions for BM25."""
    tokens = [fold(key).replace(" ", "") for key in _HYPHENATED_KEY.findall(text)]
    tokens.extend(
        word if word.isdigit() else word[:6]
        for word in fold(text).split()
        if len(word) >= 2 and word not in STOPWORDS
    )
    return tokens
