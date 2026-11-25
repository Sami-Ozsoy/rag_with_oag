"""Answer matching shared by the evaluation runner and corpus generator."""

import re
import unicodedata


def norm(text: str) -> str:
    """Case- and Turkish-insensitive, punctuation kept: '%2' must not match any '2'."""
    text = text.replace("ı", "i").replace("İ", "I")
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.casefold())


def matches(text: str, groups: list[list[str]]) -> bool:
    haystack = norm(text)
    return all(any(norm(alt) in haystack for alt in group) for group in groups)


def answer_first(text: str, groups: list[list[str]], distractors: list[str]) -> bool:
    """Require every answer group and reject distractors before the first answer mention."""
    if not matches(text, groups):
        return False
    haystack = norm(text)
    first = min(
        (pos for group in groups for alt in group if (pos := haystack.find(norm(alt))) >= 0),
        default=0,
    )
    return all(not 0 <= haystack.find(norm(distractor)) < first for distractor in distractors)
