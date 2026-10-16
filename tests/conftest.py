"""Shared fixtures: one embedded FalkorDB per test session, a fresh graph per test,
and deterministic fake models so no test touches the network."""

from __future__ import annotations

import hashlib
import json
import math
import re
import uuid
from collections import Counter
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from rag_with_oag.store.connection import GraphConnection
from rag_with_oag.store.repository import GraphRepository

_WORD = re.compile(r"\w+", re.UNICODE)


class FakeEmbedder:
    """Hashed bag of words: texts sharing words get similar vectors. Deterministic."""

    def __init__(self, dim: int = 64, model: str = "fake-embed") -> None:
        self.dim = dim
        self.model = model
        self.usage: Counter[str] = Counter()
        self.calls: list[list[str]] = []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        self.usage["embedding_tokens"] += sum(len(t.split()) for t in texts)
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for word in _WORD.findall(text.casefold()):
            h = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]


class FakeLLM:
    """Answers by citing the first source in the prompt; records every request."""

    def __init__(self, reply: str | None = None) -> None:
        self.model = "fake-llm"
        self.usage: Counter[str] = Counter()
        self.reply = reply
        self.requests: list[list[dict[str, Any]]] = []

    async def complete(self, messages: Sequence[dict[str, Any]], **kwargs: Any) -> str:
        self.requests.append(list(messages))
        self.usage["prompt_tokens"] += 100
        self.usage["completion_tokens"] += 10
        if self.reply is not None:
            return self.reply
        match = re.search(r"\[(C\d+)\]", messages[-1]["content"])
        return f"According to the sources [{match.group(1)}]." if match else "No sources."


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def connection(tmp_path_factory: pytest.TempPathFactory) -> AsyncIterator[GraphConnection]:
    conn = GraphConnection(path=tmp_path_factory.mktemp("falkordb") / "test.db")
    await conn.open()
    yield conn
    await conn.close()


@pytest.fixture
def kb_name() -> str:
    return f"kb_{uuid.uuid4().hex[:12]}"


@pytest_asyncio.fixture(loop_scope="session")
async def repo(connection: GraphConnection, kb_name: str) -> AsyncIterator[GraphRepository]:
    repository = GraphRepository(connection, kb_name)
    await repository.init(embed_model="fake-embed")
    yield repository
    await repository.drop()


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A tiny contract corpus in Turkish, with headings so section paths are exercised."""
    root = tmp_path / "docs"
    root.mkdir()
    (root / "hizmet.md").write_text(
        "# Hizmet Sözleşmesi\n\n"
        "Bu sözleşme X A.Ş. ile Y Ltd. arasında 1 Mart 2024 tarihinde imzalanmıştır.\n\n"
        "## Ödeme\n\nÜcret her ayın beşinci günü banka havalesiyle ödenir.\n\n"
        "## Fesih\n\nTaraflar otuz gün önceden yazılı bildirimle sözleşmeyi feshedebilir. "
        "Fesih bildirimi noter kanalıyla yapılır.\n",
        encoding="utf-8",
    )
    (root / "kira.txt").write_text(
        "Kira sözleşmesi. Kiracı Z A.Ş. depoyu iki yıl süreyle kiralar. "
        "Kira bedeli yıllık TÜFE oranında artırılır.",
        encoding="utf-8",
    )
    (root / "duyuru.html").write_text(
        "<html><head><title>Duyuru</title><style>p{}</style></head><body>"
        "<h1>Bakım</h1><p>Sistem cumartesi gece bakıma girecektir.</p>"
        "<script>alert(1)</script></body></html>",
        encoding="utf-8",
    )
    (root / ".hidden.md").write_text("gizli", encoding="utf-8")
    (root / "notes.bin").write_bytes(b"\x00\x01")
    return root


# --- graph fixtures ----------------------------------------------------------------

CONTRACTS_ONTOLOGY = """
name: contracts
version: "1"
entities:
  Sirket:
    description: Şirket, sözleşmelerin tarafı olan tüzel kişi
    key: vergi_no
    properties:
      vergi_no: string
      sehir: string
  Sozlesme:
    description: İki şirket arasındaki sözleşme
    properties:
      imza_tarihi: date
relations:
  TARAFIDIR:
    source: Sirket
    target: Sozlesme
    properties:
      rol: {type: enum, values: [musteri, saglayici, kiraci]}
  ORTAGIDIR:
    source: Sirket
    target: Sirket
    description: kaynak şirket, hedef şirketin ortağıdır
    properties:
      oran: number
"""

GRAPH_DOCS = {
    "hizmet.md": (
        "# HS-2024-01 Hizmet Sözleşmesi\n\n"
        "HS-2024-01 Hizmet Sözleşmesi, Atlas Lojistik A.Ş. (müşteri) ile Bulut Yazılım Ltd. "
        "(hizmet sağlayıcı) arasında 1 Mart 2024 tarihinde imzalanmıştır."
    ),
    "sicil.md": (
        "# Ticaret Sicil Kaydı\n\n"
        "Bulut Yazılım Ltd. vergi numarası 1234567890 olup merkezi Ankara'dadır. "
        "Şirketin ana ortağı Delta Holding A.Ş.'dir; ortaklık oranı yüzde 60."
    ),
    "kira.md": "# Depo Kira Sözleşmesi\n\nKiracı Zirve Gıda A.Ş. Kocaeli'deki depoyu kiralar.",
}

# name -> (type, properties stated only when the passage contains the trigger word)
ENTITY_RULES: dict[str, tuple[str, dict[str, tuple[str, Any]]]] = {
    "Atlas Lojistik A.Ş.": ("Sirket", {}),
    "Bulut Yazılım Ltd.": (
        "Sirket",
        {"vergi": ("vergi_no", "1234567890"), "Ankara": ("sehir", "Ankara")},
    ),
    "Delta Holding A.Ş.": ("Sirket", {}),
    "Zirve Gıda A.Ş.": ("Sirket", {}),
    "HS-2024-01 Hizmet Sözleşmesi": ("Sozlesme", {"Mart": ("imza_tarihi", "2024-03-01")}),
    "Depo Kira Sözleşmesi": ("Sozlesme", {}),
}
# (type, source, target, properties): emitted when both names occur in one passage
RELATION_RULES = [
    ("TARAFIDIR", "Atlas Lojistik A.Ş.", "HS-2024-01 Hizmet Sözleşmesi", {"rol": "musteri"}),
    ("TARAFIDIR", "Bulut Yazılım Ltd.", "HS-2024-01 Hizmet Sözleşmesi", {"rol": "saglayici"}),
    ("ORTAGIDIR", "Delta Holding A.Ş.", "Bulut Yazılım Ltd.", {"oran": 60}),
    ("TARAFIDIR", "Zirve Gıda A.Ş.", "Depo Kira Sözleşmesi", {"rol": "kiraci"}),
]


def fake_extraction(passages: str) -> dict[str, Any]:
    """Deterministic 'model' output for an extraction prompt, by string matching."""
    entities: dict[str, dict[str, Any]] = {}
    relations = []
    blocks = re.findall(r"\[(K\d+)\] (.*?)(?=\n\n\[K\d+\] |\Z)", passages, re.S)
    for label, text in blocks:
        present = [n for n in ENTITY_RULES if n in text]
        for name in present:
            etype, triggers = ENTITY_RULES[name]
            ent = entities.setdefault(
                name,
                {
                    "ref": f"e{len(entities) + 1}",
                    "type": etype,
                    "name": name,
                    "description": None,
                    "properties": {},
                    "chunks": [],
                },
            )
            ent["chunks"].append(label)
            for word, (prop, value) in triggers.items():
                if word in text:
                    ent["properties"][prop] = value
        for rtype, src, tgt, props in RELATION_RULES:
            if src in present and tgt in present:
                relations.append(
                    {
                        "type": rtype,
                        "source": entities[src]["ref"],
                        "target": entities[tgt]["ref"],
                        "properties": props,
                        "confidence": 0.9,
                        "chunks": [label],
                    }
                )
    return {
        "entities": list(entities.values()),
        "relations": relations,
        "unmapped": [{"term": "depozito", "count": 1}] if "depo" in passages else [],
    }


class FakeGraphLLM(FakeLLM):
    """Extracts with ``fake_extraction`` and answers by citing the first fact and chunk.

    ``unsupported`` lists structured methods (response_format types) to reject the way
    a server without that feature would.
    """

    def __init__(self, unsupported: set[str] | None = None) -> None:
        super().__init__()
        self.model = "fake-graph-llm"
        self.unsupported = unsupported or set()
        self.extraction_calls = 0

    async def complete(self, messages: Sequence[dict[str, Any]], **kwargs: Any) -> str:
        fmt = (kwargs.get("response_format") or {}).get("type")
        if fmt in self.unsupported:
            import httpx2
            import openai

            request = httpx2.Request("POST", "http://fake/v1/chat/completions")
            raise openai.BadRequestError(
                f"{fmt} not supported", response=httpx2.Response(400, request=request), body=None
            )
        passages = next((m["content"] for m in messages if "Passages:" in m["content"]), None)
        if passages is not None:
            self.requests.append(list(messages))
            self.extraction_calls += 1
            return json.dumps(fake_extraction(passages), ensure_ascii=False)
        self.requests.append(list(messages))
        cites = re.findall(r"\[(F1|C1)\]", messages[-1]["content"])
        return "Answer " + "".join(f"[{c}]" for c in dict.fromkeys(cites)) + "."


@pytest.fixture
def ontology() -> Any:
    from rag_with_oag import Ontology

    return Ontology.from_yaml(CONTRACTS_ONTOLOGY)


@pytest.fixture
def graph_docs() -> list[Any]:
    from rag_with_oag import Document

    return [Document.from_text(text, source=name) for name, text in GRAPH_DOCS.items()]
