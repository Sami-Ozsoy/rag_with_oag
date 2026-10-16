"""M0 spike: is FalkorDB good enough to build on?

Measures, on embedded FalkorDB (falkordblite):
  1. ANN recall@k of the HNSW vector index against exact numpy search
  2. p50/p95 latency of a rag vector query and of the hybrid prototype query
     (entity linking -> 2-hop expansion -> evidence chunks), LLM excluded
  3. memory use against chunk count
  4. the direction of the vector score (similarity or distance)

Default vectors are synthetic but clustered, like real embeddings. Pass
``--embed fastembed --corpus DIR`` to run on real text instead.

    python spikes/m0_falkordb.py                          # ~3k chunks, 384-dim
    python spikes/m0_falkordb.py --chunks 4000 --dim 768
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from rag_with_oag.store.connection import GraphConnection
from rag_with_oag.store.schema import VECTOR_EF_RUNTIME
from rag_with_oag.types import Chunk, Document, chunk_id

# Proposed gate. The plan leaves the numbers open; these are a starting point to agree on.
GATE = {"recall_at_k": 0.95, "rag_p95_ms": 50.0, "hybrid_p95_ms": 150.0}

CHUNK_SEARCH = (
    "CALL db.idx.vector.queryNodes('Chunk', 'embedding', $k, vecf32($q)) "
    "YIELD node, score RETURN node.id, score"
)
HYBRID = """
CALL db.idx.vector.queryNodes('Entity', 'embedding', $k_over, vecf32($q)) YIELD node AS e, score
WITH e, score WHERE any(l IN labels(e) WHERE l IN $types)
WITH e, score ORDER BY score ASC LIMIT $k_entities
MATCH p = (e)-[:TARAFIDIR|ICERIR*1..2]-(n:Entity)
WITH collect(DISTINCT e.id) + collect(DISTINCT n.id) AS ids, collect(p)[..$max_paths] AS paths
MATCH (c:Chunk)-[:MENTIONS]->(x:Entity) WHERE x.id IN ids
WITH DISTINCT c, paths
RETURN c.id, vec.cosineDistance(c.embedding, vecf32($q)) AS dist, size(paths)
ORDER BY dist ASC LIMIT $k
"""


@dataclass
class Result:
    platform: str
    chunks: int
    entities: int
    dim: int
    ef: int
    k: int
    queries: int
    ingest_s: float
    recall_at_k: float
    recall_min: float
    rag_p50_ms: float
    rag_p95_ms: float
    hybrid_p50_ms: float
    hybrid_p95_ms: float
    hybrid_avg_hits: float
    memory_mb: float
    bytes_per_chunk: float
    score_is_distance: bool
    gate: dict[str, bool]


def clustered_vectors(n: int, dim: int, rng: np.random.Generator, clusters: int = 40) -> np.ndarray:
    centers = rng.normal(size=(clusters, dim))
    labels = rng.integers(0, clusters, size=n)
    vecs = centers[labels] + rng.normal(scale=0.6, size=(n, dim))
    return (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)).astype(np.float32)


async def embed_corpus(args: argparse.Namespace) -> tuple[list[str], np.ndarray]:
    from rag_with_oag.embed.local import FastEmbedEmbedder
    from rag_with_oag.ingest.chunker import AdaptiveChunker
    from rag_with_oag.ingest.loaders import iter_paths, load_path

    chunker = AdaptiveChunker()
    texts = [s.text for p in iter_paths([args.corpus]) for s in chunker.split(load_path(p).text)]
    texts = texts[: args.chunks]
    vectors = await FastEmbedEmbedder().embed(texts)
    return texts, np.asarray(vectors, dtype=np.float32)


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


async def run(args: argparse.Namespace) -> Result:
    rng = np.random.default_rng(args.seed)
    if args.embed == "fastembed":
        texts, vectors = await embed_corpus(args)
    else:
        vectors = clustered_vectors(args.chunks, args.dim, rng)
        texts = [f"synthetic chunk {i}" for i in range(len(vectors))]
    n, dim = vectors.shape

    workdir = Path(tempfile.mkdtemp(prefix="m0_"))
    conn = GraphConnection(path=workdir / "m0.db")
    await conn.open()
    graph = conn.graph("m0")
    try:
        await graph.query(
            f"CREATE VECTOR INDEX FOR (c:Chunk) ON (c.embedding) "
            f"OPTIONS {{dimension: {dim}, similarityFunction: 'cosine', efRuntime: {args.ef}}}"
        )
        await graph.query(
            f"CREATE VECTOR INDEX FOR (e:Entity) ON (e.embedding) "
            f"OPTIONS {{dimension: {dim}, similarityFunction: 'cosine', efRuntime: {args.ef}}}"
        )
        await graph.query("CREATE INDEX FOR (c:Chunk) ON (c.id)")
        await graph.query("CREATE INDEX FOR (e:Entity) ON (e.id)")

        # --- ingest chunks in ~300-500 "documents" ------------------------------
        t0 = time.perf_counter()
        per_doc = max(1, n // 400)
        rows = []
        for i, (text, vec) in enumerate(zip(texts, vectors, strict=True)):
            doc = Document.from_text(f"doc {i // per_doc}", source=f"doc{i // per_doc}")
            c = Chunk(chunk_id(doc.id, i), doc.id, i, text, 0, 0, len(text))
            rows.append({"id": c.id, "doc": doc.id, "text": text, "e": vec.tolist()})
        ids = [r["id"] for r in rows]
        for i in range(0, len(rows), 256):
            await graph.query(
                "UNWIND $rows AS r MERGE (d:Document {id: r.doc}) "
                "CREATE (d)-[:HAS_CHUNK]->"
                "(:Chunk {id: r.id, text: r.text, embedding: vecf32(r.e)})",
                {"rows": rows[i : i + 256]},
            )

        # --- a small hand-made ontology: Sirket -TARAFIDIR-> Sozlesme -ICERIR-> Madde
        e_count = args.entities
        e_vecs = clustered_vectors(e_count, dim, rng)
        types = ["Sirket", "Sozlesme", "Madde"]
        for i in range(e_count):
            label = types[i % 3]
            await graph.query(
                f"CREATE (:Entity:{label} {{id: $id, name: $name, embedding: vecf32($e)}})",
                {"id": f"e{i}", "name": f"{label} {i}", "e": e_vecs[i].tolist()},
            )
        rel = []
        for i in range(0, e_count - 2, 3):
            rel.append({"a": f"e{i}", "b": f"e{i + 1}", "t": "TARAFIDIR"})
            rel.append({"a": f"e{i + 1}", "b": f"e{i + 2}", "t": "ICERIR"})
        for t in ("TARAFIDIR", "ICERIR"):
            await graph.query(
                f"UNWIND $rows AS r MATCH (a:Entity {{id: r.a}}), (b:Entity {{id: r.b}}) "
                f"CREATE (a)-[:{t} {{evidence: []}}]->(b)",
                {"rows": [r for r in rel if r["t"] == t]},
            )
        mentions = [
            {"c": ids[int(rng.integers(n))], "e": f"e{int(rng.integers(e_count))}"}
            for _ in range(min(n, e_count * 20))
        ]
        await graph.query(
            "UNWIND $rows AS r MATCH (c:Chunk {id: r.c}), (e:Entity {id: r.e}) "
            "CREATE (c)-[:MENTIONS {confidence: 0.9}]->(e)",
            {"rows": mentions},
        )
        ingest_s = time.perf_counter() - t0

        # --- recall and latency ---------------------------------------------------
        queries = clustered_vectors(args.queries, dim, rng)
        recalls, rag_ms, hybrid_ms, hybrid_hits = [], [], [], []
        distance_like = True
        for qv in queries:
            exact = set(np.argsort(-(vectors @ qv))[: args.k].tolist())
            t = time.perf_counter()
            res = await graph.ro_query(CHUNK_SEARCH, {"k": args.k, "q": qv.tolist()})
            rag_ms.append((time.perf_counter() - t) * 1000)
            got = {ids.index(r[0]) for r in res.result_set}
            recalls.append(len(got & exact) / args.k)
            scores = [r[1] for r in res.result_set]
            distance_like &= scores == sorted(scores)  # ascending => distance

            t = time.perf_counter()
            res = await graph.ro_query(
                HYBRID,
                {
                    "q": qv.tolist(),
                    "k": args.k,
                    "k_entities": 5,
                    "k_over": 15,
                    "types": ["Sirket", "Sozlesme"],
                    "max_paths": 50,
                },
            )
            hybrid_ms.append((time.perf_counter() - t) * 1000)
            hybrid_hits.append(len(res.result_set))

        # the best match for a stored vector must be itself with score ~0
        probe = await graph.ro_query(CHUNK_SEARCH, {"k": 1, "q": vectors[0].tolist()})
        score_is_distance = distance_like and abs(probe.result_set[0][1]) < 1e-3

        info = await conn._client.connection.info("memory")  # type: ignore[union-attr]
        memory_mb = info["used_memory"] / 1e6
    finally:
        await conn.close()

    recall = statistics.fmean(recalls)
    result = Result(
        platform=f"{platform.system()} {platform.machine()} / Python {sys.version.split()[0]}",
        chunks=n,
        entities=args.entities,
        dim=dim,
        ef=args.ef,
        k=args.k,
        queries=args.queries,
        ingest_s=round(ingest_s, 2),
        recall_at_k=round(recall, 4),
        recall_min=round(min(recalls), 2),
        rag_p50_ms=round(pct(rag_ms, 50), 2),
        rag_p95_ms=round(pct(rag_ms, 95), 2),
        hybrid_p50_ms=round(pct(hybrid_ms, 50), 2),
        hybrid_p95_ms=round(pct(hybrid_ms, 95), 2),
        hybrid_avg_hits=round(statistics.fmean(hybrid_hits), 1),
        memory_mb=round(memory_mb, 1),
        bytes_per_chunk=round(memory_mb * 1e6 / n),
        score_is_distance=score_is_distance,
        gate={},
    )
    result.gate = {
        "recall_at_k": result.recall_at_k >= GATE["recall_at_k"],
        "rag_p95_ms": result.rag_p95_ms <= GATE["rag_p95_ms"],
        "hybrid_p95_ms": result.hybrid_p95_ms <= GATE["hybrid_p95_ms"],
    }
    return result


def report(r: Result) -> str:
    direction = "cosine distance (lower = closer)" if r.score_is_distance else "similarity"

    def mark(key: str) -> str:
        return "✅" if r.gate[key] else "❌"

    return "\n".join(
        [
            f"# M0 FalkorDB spike — {r.platform}",
            "",
            f"{r.chunks} chunks × {r.dim} dim, efRuntime {r.ef}, {r.entities} entities, k={r.k}, "
            f"{r.queries} queries; "
            f"ingest {r.ingest_s}s",
            "",
            "| Metric | Value | Gate |",
            "|---|---|---|",
            f"| ANN recall@{r.k} (mean / min) | {r.recall_at_k} / {r.recall_min} "
            f"| ≥ {GATE['recall_at_k']} {mark('recall_at_k')} |",
            f"| rag query p50 / p95 | {r.rag_p50_ms} / {r.rag_p95_ms} ms "
            f"| p95 ≤ {GATE['rag_p95_ms']} ms {mark('rag_p95_ms')} |",
            f"| hybrid query p50 / p95 | {r.hybrid_p50_ms} / {r.hybrid_p95_ms} ms "
            f"| p95 ≤ {GATE['hybrid_p95_ms']} ms {mark('hybrid_p95_ms')} |",
            f"| hybrid evidence chunks (avg) | {r.hybrid_avg_hits} | |",
            f"| server memory | {r.memory_mb} MB (~{r.bytes_per_chunk} B/chunk) | |",
            f"| score direction | {direction} | |",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunks", type=int, default=3000)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--entities", type=int, default=90)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ef", type=int, default=VECTOR_EF_RUNTIME, help="HNSW efRuntime")
    parser.add_argument("--embed", choices=["synthetic", "fastembed"], default="synthetic")
    parser.add_argument("--corpus", type=Path, help="directory of documents for --embed fastembed")
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "out")
    args = parser.parse_args()
    if args.embed == "fastembed" and not args.corpus:
        parser.error("--embed fastembed needs --corpus")

    result = asyncio.run(run(args))
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"m0_{result.chunks}x{result.dim}_ef{result.ef}"
    (args.out / f"{stem}.json").write_text(json.dumps(asdict(result), indent=2))
    text = report(result)
    (args.out / f"{stem}.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
