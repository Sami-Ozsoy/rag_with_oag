"""Text folding for matching names across case, diacritics and Turkish dotted/dotless i.

FalkorDB's full-text index does not case-fold non-ASCII letters ("şimşek" does
not find "Şimşek"), so entity names are stored and queried in folded form.
"""

from __future__ import annotations

import re
import unicodedata

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

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


def fulltext_query(question: str, *, prefix_len: int = 6) -> str:
    """OR-query of folded question terms, prefix-matched to absorb Turkish suffixes.

    'Atlas Lojistik'in sözleşmeleri' -> 'atlas*|lojist*|sozles*'. Terms are pure
    [a-z0-9], so nothing in the question can inject query syntax.
    """
    terms: list[str] = []
    for word in fold(question).split():
        if len(word) < 3 or word in STOPWORDS:
            continue
        term = (word[:prefix_len] if len(word) > prefix_len else word) + "*"
        if term not in terms:
            terms.append(term)
    return "|".join(terms)
