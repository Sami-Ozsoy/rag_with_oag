"""Turn paths into Documents. Supports txt, md, rst, html and (with the pdf extra) pdf."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from html.parser import HTMLParser
from pathlib import Path
from typing import ClassVar

from rag_with_oag.types import Document

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".rst"}
HTML_SUFFIXES = {".html", ".htm"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | HTML_SUFFIXES | PDF_SUFFIXES

Source = str | Path | Document


def iter_paths(sources: Iterable[str | Path]) -> Iterator[Path]:
    """Expand directories recursively; skip hidden files and unsupported types."""
    for source in sources:
        path = Path(source).expanduser()
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                rel = child.relative_to(path)
                if (
                    child.is_file()
                    and child.suffix.lower() in SUPPORTED_SUFFIXES
                    and not any(part.startswith(".") for part in rel.parts)
                ):
                    yield child
        elif path.is_file():
            yield path
        else:
            raise FileNotFoundError(f"no such file or directory: {path}")


def load_path(path: Path) -> Document:
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        text = path.read_text(encoding="utf-8")
        return Document(
            text=text,
            source=str(path),
            title=_markdown_title(text) or path.stem,
            metadata={"format": suffix.lstrip(".")},
        )
    if suffix in HTML_SUFFIXES:
        parser = _HTMLText()
        parser.feed(path.read_text(encoding="utf-8", errors="replace"))
        return Document(
            text=parser.text(),
            source=str(path),
            title=parser.title or path.stem,
            metadata={"format": "html"},
        )
    if suffix in PDF_SUFFIXES:
        return _load_pdf(path)
    raise ValueError(
        f"unsupported file type '{suffix}' ({path}); supported: {sorted(SUPPORTED_SUFFIXES)}"
    )


def _load_pdf(path: Path) -> Document:
    try:
        from adaptive_chunking.io import load_pdf_file
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PDF support needs: pip install 'rag_with_oag[pdf]'") from exc
    try:
        loaded = load_pdf_file(path)
    except RuntimeError as exc:  # raised by adaptive_chunking when pypdf is missing
        raise ImportError("PDF support needs: pip install 'rag_with_oag[pdf]'") from exc
    # Pages are joined with form feeds; the chunker turns those into page indexes.
    return Document(
        text=loaded.text,
        source=str(path),
        title=path.stem,
        metadata={"format": "pdf", "page_count": loaded.metadata.get("page_count")},
    )


def _markdown_title(text: str) -> str | None:
    match = re.search(r"^#\s+(.+?)\s*#*\s*$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


class _HTMLText(HTMLParser):
    """Visible text of an HTML page, with headings kept as markdown for section paths."""

    _SKIP: ClassVar[frozenset[str]] = frozenset(
        {"script", "style", "noscript", "template", "svg", "head"}
    )
    _BLOCK: ClassVar[frozenset[str]] = frozenset(
        {
            "p",
            "div",
            "section",
            "article",
            "li",
            "tr",
            "br",
            "table",
            "ul",
            "ol",
            "pre",
            "blockquote",
            "header",
            "footer",
            "main",
            "nav",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False
        self.title: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        elif tag in self._SKIP:
            self._skip_depth += 1
        elif re.fullmatch(r"h[1-6]", tag):
            self._parts.append("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif re.fullmatch(r"h[1-6]", tag) or tag in self._BLOCK:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title = (self.title or "") + data.strip()
        elif not self._skip_depth:
            self._parts.append(data)

    def text(self) -> str:
        joined = "".join(self._parts)
        joined = re.sub(r"[ \t]+", " ", joined)
        joined = re.sub(r" *\n *", "\n", joined)
        return re.sub(r"\n{3,}", "\n\n", joined).strip()
