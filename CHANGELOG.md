# Changelog

## [Unreleased]

## [0.2.0] - 2026-11-26

First release on PyPI. No changes since 0.2.0rc1.

## [0.2.0rc1] - 2026-11-26

### Added

- Readable model errors for connection failures, authentication and missing models (`ModelConnectionError`, `ModelAuthError` and `ModelNotFoundError`).
- A two-document, three-hop [quick start example](examples/quickstart/).
- `Ontology.suggest()` and the `suggest` CLI command to draft and extend ontologies.
- Re-ingest changed files and delete documents while keeping relations with other evidence.
- Local cross-encoder reranking and query and document prefixes for embeddings.
- `reasoning_effort` for OpenAI-compatible models.
- Held-out synthetic Turkish contracts and a converter for 2WikiMultiHopQA, with equal-budget comparisons and gold evidence metrics.

### Changed

- RAG and hybrid fuse BM25 keyword ranks with vector ranks by default.
- Hybrid reserves top text passages before packing graph facts. Graph traversal keeps nearer facts first.

### Fixed

- Match exact entity names even when full-text search leaves them out of its candidate pool.
- Count refusals as wrong in evaluation, including on choice questions.
- Raise `OutputTruncatedError` when a model reaches its limit before writing an answer.

## [0.2.0.dev0] - 2026-10-16

### Added

- RAG, ontology-augmented and hybrid modes over one FalkorDB graph, with fact and passage citations.
- Ontology-based extraction, validation and entity resolution, with a CLI for ingestion and queries.
- A golden evaluation set with 14 contract documents and 33 questions.
