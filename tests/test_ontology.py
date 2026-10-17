"""Ontology model, schema generation, extraction validation, structured output, query safety."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from typing import Any, Literal

import httpx2
import pytest
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from rag_with_oag import LLMConfig, Ontology, OntologyError, Property, StructuredOutputError
from rag_with_oag._text import fold, fulltext_query
from rag_with_oag.ingest.extractor import validate
from rag_with_oag.llm import capabilities, structured
from rag_with_oag.llm.capabilities import StructuredMethod
from rag_with_oag.llm.client import LLMClient
from rag_with_oag.ontology.schema_gen import RawExtraction, extraction_schema
from rag_with_oag.ontology.validator import coerce_properties
from rag_with_oag.retrieval.linking import rrf
from rag_with_oag.store import queries as q

from .conftest import CONTRACTS_ONTOLOGY, FakeGraphLLM, fake_extraction


@pytest.fixture(autouse=True)
def _fresh_capabilities() -> None:
    capabilities.forget()


# --- ontology model ------------------------------------------------------------


def test_yaml_round_trip_keeps_meaning_and_fingerprint(ontology: Ontology) -> None:
    again = Ontology.from_yaml(ontology.to_yaml())
    assert again == ontology and again.fingerprint == ontology.fingerprint
    sirket = ontology.entity("Sirket")
    assert sirket is not None and sirket.key == "vergi_no"
    rol = ontology.relation("TARAFIDIR").property("rol")  # type: ignore[union-attr]
    assert rol is not None and rol.values == ("musteri", "saglayici", "kiraci")
    assert "TARAFIDIR: Sirket -> Sozlesme" in ontology.describe()


def test_fingerprint_changes_with_content(ontology: Ontology) -> None:
    changed = Ontology.from_yaml(CONTRACTS_ONTOLOGY.replace("sehir: string", "il: string"))
    assert changed.fingerprint != ontology.fingerprint


def test_from_pydantic_classes() -> None:
    class Sirket(BaseModel):
        """Tüzel kişi."""

        __ontology_key__ = "vergi_no"
        vergi_no: str | None = None
        tur: Literal["anonim", "limited"] | None = None
        calisan_sayisi: int = Field(description="toplam çalışan")

    class Sozlesme(BaseModel):
        imza_tarihi: str | None = None

    class Taraflik(BaseModel):
        """Şirket sözleşmenin tarafıdır."""

        rol: Literal["musteri", "saglayici"]

    o = Ontology.from_classes([Sirket, Sozlesme], {"TARAFIDIR": (Sirket, Sozlesme, Taraflik)})
    s = o.entity("Sirket")
    assert s is not None and s.description == "Tüzel kişi." and s.key == "vergi_no"
    props = {p.name: p for p in s.properties}
    assert props["tur"].type == "enum" and props["tur"].values == ("anonim", "limited")
    assert props["calisan_sayisi"].required and props["calisan_sayisi"].type == "integer"
    assert not props["vergi_no"].required
    rel = o.relation("TARAFIDIR")
    assert rel is not None and rel.description.startswith("Şirket")
    assert rel.properties[0].required


@pytest.mark.parametrize(
    ("yaml", "message"),
    [
        ("entities:\n  Şirket: {}\n", "ASCII identifiers"),
        ("entities:\n  Chunk: {}\n", "reserved"),
        ("entities:\n  A: {key: x}\n", "key 'x' is not one of its properties"),
        ("entities:\n  A: {properties: {t: enum}}\n", "needs 'values'"),
        ("entities:\n  A: {properties: {name: string}}\n", "reserved"),
        (
            "entities:\n  A: {}\nrelations:\n  R: {source: A, target: B}\n",
            "unknown entity type 'B'",
        ),
        ("entities:\n  A: {}\nrelations:\n  MENTIONS: {source: A, target: A}\n", "reserved"),
        ("entities: {}\n", "at least one entity type"),
        ("entities:\n  A: {}\nfoo: 1\n", "unknown top-level"),
    ],
)
def test_invalid_ontologies_are_explained(yaml: str, message: str) -> None:
    with pytest.raises(OntologyError, match=message):
        Ontology.from_yaml(yaml)


# --- schema --------------------------------------------------------------------


def _assert_strict(schema: dict[str, Any], path: str = "$") -> None:
    """OpenAI strict mode: every object closed, every property required."""
    if schema.get("type") == "object":
        assert schema.get("additionalProperties") is False, path
        assert sorted(schema.get("required", [])) == sorted(schema.get("properties", {})), path
    for key in ("properties",):
        for name, sub in schema.get(key, {}).items():
            _assert_strict(sub, f"{path}.{name}")
    if "items" in schema:
        _assert_strict(schema["items"], f"{path}[]")
    for variant in schema.get("anyOf", []):
        _assert_strict(variant, f"{path}|")


def test_extraction_schema_is_strict_and_typed(ontology: Ontology) -> None:
    schema = extraction_schema(ontology)
    _assert_strict(schema)
    variants = schema["properties"]["entities"]["items"]["anyOf"]
    sirket = next(v for v in variants if v["properties"]["type"]["enum"] == ["Sirket"])
    assert set(sirket["properties"]["properties"]["properties"]) == {"vergi_no", "sehir"}
    rel = schema["properties"]["relations"]["items"]["anyOf"][0]
    rol = rel["properties"]["properties"]["properties"]["rol"]
    assert rol["enum"] == ["musteri", "saglayici", "kiraci", None]


# --- property coercion and extraction validation -------------------------------


def test_coerce_properties(ontology: Ontology) -> None:
    tarafidir = ontology.relation("TARAFIDIR")
    assert tarafidir is not None
    clean, errors = coerce_properties(tarafidir.properties, {"rol": "Musteri", "x": 1}, "r")
    assert clean == {"rol": "musteri"} and not errors
    _, errors = coerce_properties(tarafidir.properties, {"rol": "ortak"}, "r")
    assert errors and "not one of" in errors[0]
    sozlesme = ontology.entity("Sozlesme")
    assert sozlesme is not None
    clean, errors = coerce_properties(sozlesme.properties, {"imza_tarihi": "2024-03-01T00:00"}, "e")
    assert clean == {"imza_tarihi": "2024-03-01"}
    _, errors = coerce_properties(sozlesme.properties, {"imza_tarihi": "1 Mart 2024"}, "e")
    assert errors and "YYYY-MM-DD" in errors[0]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (3, 3),
        (3.0, 3),
        ("42", 42),
        ("1.500", 1500),
        ("1,500", 1500),
        ("1.500.000", 1500000),
        ("1 500 000", 1500000),
        ("1.500.000,00", 1500000),
        (2.5, None),
        ("12,5", None),
        ("on iki", None),
        (True, None),
    ],
)
def test_coerce_integer(value: Any, expected: int | None) -> None:
    props = (Property(name="adet", type="integer"),)
    clean, errors = coerce_properties(props, {"adet": value}, "e")
    if expected is None:
        assert errors and not clean
    else:
        assert clean == {"adet": expected} and not errors
        assert type(clean["adet"]) is int


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (60, 60.0),
        (2.5, 2.5),
        ("12,5", 12.5),
        ("12.5", 12.5),
        ("%40", 40.0),
        ("% 40", 40.0),
        ("0.125", 0.125),
        ("1.500", 1500.0),
        ("1.500.000,50", 1500000.5),
        ("1,500,000.50", 1500000.5),
        ("-3,75", -3.75),
        ("yüzde altmış", None),
        ("NaN", None),
    ],
)
def test_coerce_number(value: Any, expected: float | None) -> None:
    props = (Property(name="oran", type="number"),)
    clean, errors = coerce_properties(props, {"oran": value}, "e")
    if expected is None:
        assert errors and not clean
    else:
        assert clean == {"oran": expected} and not errors


def test_validate_splits_per_chunk_and_reports_errors(ontology: Ontology) -> None:
    raw = RawExtraction.model_validate(
        {
            "entities": [
                {"ref": "e1", "type": "Sirket", "name": "Atlas", "chunks": ["K1"]},
                {"ref": "e2", "type": "Sozlesme", "name": "HS-1", "chunks": ["K1", "K2"]},
                {"ref": "e3", "type": "Kisi", "name": "Ali", "chunks": ["K1"]},
                {"ref": "e4", "type": "Sirket", "name": "Bulut", "chunks": ["K9"]},
            ],
            "relations": [
                {
                    "type": "TARAFIDIR",
                    "source": "e1",
                    "target": "e2",
                    "chunks": [],
                    "properties": {"rol": "musteri"},
                },
                {"type": "TARAFIDIR", "source": "e2", "target": "e1", "chunks": ["K1"]},
                {"type": "ORTAGIDIR", "source": "e1", "target": "e3", "chunks": ["K1"]},
                {"type": "TARAFIDIR", "source": "e1", "target": "e2", "chunks": ["K2"]},
            ],
            "unmapped": [{"term": "teminat", "count": 2}],
        }
    )
    graphs, errors = validate(raw, {"K1": "c1", "K2": "c2"}, ontology)
    assert any("unknown entity type 'Kisi'" in e for e in errors)
    assert any("unknown passage id 'K9'" in e for e in errors)
    assert any("connects Sirket -> Sozlesme, not Sozlesme -> Sirket" in e for e in errors)
    assert any("refs of valid entities" in e for e in errors)
    # relation without chunks: evidence = the passage mentioning both ends (K1)
    assert [r.type for r in graphs["c1"].relations] == ["TARAFIDIR"]
    # relation stated in K2 carries Atlas along without claiming K2 mentions it
    atlas_in_c2 = graphs["c2"].entity("e1")
    assert atlas_in_c2 is not None and not atlas_in_c2.mentioned
    assert graphs["c1"].unmapped == [("teminat", 2)]


# --- structured output ---------------------------------------------------------


def _extraction_messages() -> list[dict[str, Any]]:
    return [{"role": "user", "content": "Passages:\n\n[K1] Atlas Lojistik A.Ş. burada."}]


async def test_structured_falls_back_and_remembers(ontology: Ontology) -> None:
    llm = FakeGraphLLM(unsupported={"json_schema"})
    schema = extraction_schema(ontology)
    first = await structured.generate(
        llm, _extraction_messages(), schema=schema, model=RawExtraction
    )
    # json_schema rejected, the fake has no tool calling, json_object works
    assert first.method == StructuredMethod.JSON_OBJECT
    assert first.value.entities[0].name == "Atlas Lojistik A.Ş."
    assert capabilities.detected(llm) == StructuredMethod.JSON_OBJECT
    llm.requests.clear()
    await structured.generate(llm, _extraction_messages(), schema=schema, model=RawExtraction)
    assert len(llm.requests) == 1  # went straight to the remembered method


class _ScriptedLLM:
    """Replies with the given strings in order, recording what it was sent."""

    def __init__(self, replies: Sequence[str]) -> None:
        self.model = "scripted"
        self.usage: Counter[str] = Counter()
        self.replies = list(replies)
        self.sent: list[list[dict[str, Any]]] = []

    async def complete(self, messages: Sequence[dict[str, Any]], **kwargs: Any) -> str:
        self.sent.append(list(messages))
        return self.replies.pop(0)


async def test_structured_retries_with_validation_feedback() -> None:
    good = json.dumps({"entities": [], "relations": [], "unmapped": []})
    llm = _ScriptedLLM(['{"entities": "oops"}', f"```json\n{good}\n```"])
    result = await structured.generate(
        llm, [{"role": "user", "content": "x"}], schema={}, model=RawExtraction, mode="json_schema"
    )
    assert result.attempts == 2
    assert "That JSON has problems" in llm.sent[1][-1]["content"]


async def test_structured_gives_up_with_clear_error() -> None:
    llm = _ScriptedLLM(["not json at all"] * 4)
    with pytest.raises(StructuredOutputError, match="tried"):
        await structured.generate(
            llm, [{"role": "user", "content": "x"}], schema={}, model=RawExtraction
        )


def _sdk(handler: Any) -> tuple[AsyncOpenAI, list[httpx2.Request]]:
    seen: list[httpx2.Request] = []

    def record(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return handler(request)

    return AsyncOpenAI(
        base_url="http://llm.test/v1",
        api_key="k",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(record)),
    ), seen


def _reply(message: dict[str, Any]) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        },
    )


async def test_real_client_json_schema_then_tools(ontology: Ontology) -> None:
    payload = fake_extraction("[K1] Atlas Lojistik A.Ş. burada.")

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        if "response_format" in body:
            return httpx2.Response(400, json={"error": {"message": "json_schema unsupported"}})
        assert body["tool_choice"]["function"]["name"] == "extraction"
        assert body["tools"][0]["function"]["strict"] is True
        return _reply(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "extraction", "arguments": json.dumps(payload)},
                    }
                ],
            }
        )

    sdk, seen = _sdk(handler)
    client = LLMClient(LLMConfig(model="m", base_url="http://llm.test/v1"), client=sdk)
    result = await structured.generate(
        client,
        _extraction_messages(),
        schema=extraction_schema(ontology),
        model=RawExtraction,
        name="extraction",
    )
    assert result.method == StructuredMethod.TOOLS
    first = json.loads(seen[0].content)
    assert first["response_format"]["json_schema"]["strict"] is True
    await client.aclose()


# --- text folding, fusion, query safety -----------------------------------------


def test_fold_handles_turkish_case_and_diacritics() -> None:
    assert fold("ŞİMŞEK İnşaat A.Ş.") == fold("şimşek insaat a s") == "simsek insaat a s"
    assert fold("IŞIK") == fold("ışık")


def test_fulltext_query_is_prefix_terms_without_syntax() -> None:
    assert fulltext_query("Atlas Lojistik'in sözleşmeleri nedir?") == "atlas*|lojist*|sozles*"
    assert fulltext_query("ne ve ile") == ""
    hostile = fulltext_query('x\') | @field:{*} -foo ~bar %baz% "q"')
    assert set(hostile) <= set("abcdefghijklmnopqrstuvwxyz0123456789*|")


def test_rrf_rewards_agreement() -> None:
    fused = rrf([["a", "b", "c"], ["c", "a"]], key=lambda x: x)
    assert [item for item, _ in fused] == ["a", "c", "b"]


@pytest.mark.parametrize("bad", ["TARAFIDIR]->(x) DETACH DELETE x //", "A|B", "1X"])
def test_query_builders_reject_unsafe_identifiers(bad: str) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        q.expand([bad], 2)
    with pytest.raises(ValueError, match="unsafe"):
        q.upsert_entities(bad)
    with pytest.raises(ValueError, match="unsafe"):
        q.find_by_key("Sirket", bad)


@pytest.mark.parametrize("hops", [0, 6, "2; MATCH (n) DELETE n", True, 2.5])
def test_expand_rejects_bad_hops(hops: Any) -> None:
    with pytest.raises(ValueError, match="hops"):
        q.expand(["TARAFIDIR"], hops)
