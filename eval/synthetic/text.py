"""Answer matching shared by the evaluation runner and corpus generator."""

import re
import unicodedata
from collections.abc import Iterable
from datetime import date

_APOSTROPHES = str.maketrans(dict.fromkeys("’‘ʼ´`", "'"))
_MONTHS = (
    "January February March April May June July August September October November December"
).split()

_VERDICT_PATTERNS = (
    r"(?:cannot|can['’]t|can\s+not|unable\s+to)\s+"
    r"(?:determine|answer|tell|say|establish|compare|conclude)",
    r"(?:cannot|can['’]t|can\s+not)\s+be\s+(?:determined|established|answered)",
    r"not\s+possible\s+to\s+(?:determine|say|tell|establish|answer)",
    r"(?:does|do)\s+not\s+allow\s+(?:a\s+|us\s+to\s+)?(?:determin\w*|conclu\w*|answer\w*)",
    r"(?:does|do)\s+not\s+(?:determine|establish)\s+(?:this|it|which|who|whether)",
    r"(?:not\s+enough|insufficient)\s+(?:information|evidence)\s+to",
    r"(?:insufficient|not\s+enough|not\s+sufficient)\s+to\s+"
    r"(?:determine|answer|tell|establish|say|conclude)",
    r"no\s+way\s+to\s+(?:determine|tell)",
)
_DETAIL_PATTERNS = (
    r"(?:do|does|did)\s+not\s+(?:include|contain|provide|state|mention|say|specify|give|have)",
    r"(?:don['’]t|doesn['’]t|didn['’]t)\s+"
    r"(?:include|contain|provide|state|mention|say|specify|give|have)",
    r"(?:is|are)\s+not\s+(?:provided|stated|given|mentioned|specified|available)",
    r"(?:not\s+enough|insufficient)\s+(?:information|evidence)",
    r"no\s+(?:information|facts|evidence|sources|data)\s+(?:about|on|for|regarding)",
    r"there\s+is\s+no\s+(?:information|evidence|fact)",
)
_VERDICT_RE = re.compile(r"\b(?:" + "|".join(_VERDICT_PATTERNS) + r")\b", re.IGNORECASE)
_DETAIL_RE = re.compile(r"\b(?:" + "|".join(_DETAIL_PATTERNS) + r")\b", re.IGNORECASE)


def is_refusal(text: str, ignore: Iterable[str] = ()) -> bool:
    """Detect verdict phrases anywhere and missing-detail phrases in the first sentence."""
    for option in sorted(filter(None, ignore), key=len, reverse=True):
        text = re.sub(re.escape(option), "", text, flags=re.IGNORECASE)
    if _VERDICT_RE.search(text):
        return True
    text = re.sub(r"^[\s*_]+", "", text)
    first_sentence = re.split(r"[.!?](?=\s|$)|[\r\n]", text, maxsplit=1)[0]
    return _DETAIL_RE.search(first_sentence) is not None


def norm(text: str) -> str:
    """Fold case, diacritics and apostrophes while retaining punctuation."""
    text = text.translate(_APOSTROPHES).replace("ı", "i").replace("İ", "I")
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.casefold())


def date_forms(text: str) -> list[str]:
    """Return day-first, month-first and ISO forms of a valid English date."""
    text = " ".join(text.split()).lower()
    if match := re.fullmatch(r"([0-9]{1,2}) ([a-z]+) ([0-9]{4})", text):
        day, month, year = match.groups()
    elif match := re.fullmatch(r"([a-z]+) ([0-9]{1,2}), ([0-9]{4})", text):
        month, day, year = match.groups()
    else:
        return []

    month = month.capitalize()
    try:
        value = date(int(year), _MONTHS.index(month) + 1, int(day))
    except ValueError:
        return []
    return [f"{value.day} {month} {year}", f"{month} {value.day}, {year}", value.isoformat()]


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
