"""Exact name lookup beyond the full-text pool."""

from collections.abc import Sequence
from unittest.mock import create_autospec

import pytest

from rag_with_oag import RetrievalConfig
from rag_with_oag._text import fold, fulltext_query
from rag_with_oag.retrieval.linking import EntityLinker
from rag_with_oag.store.repository import GraphRepository
from rag_with_oag.types import EntityHit

FIRST_FILM = "Kill Me Again"
SECOND_FILM = "The Woman Dressed As A Man"
JANE = "Jane Pierce"
HOWARD = "Howard Hawks"


@pytest.mark.parametrize(
    ("question", "stored", "pool", "expected"),
    [
        pytest.param(
            "Did Jane Pierce and Howard Hawks meet?",
            [HOWARD, JANE],
            [HOWARD],
            [JANE, HOWARD],
            id="outside-fulltext-pool",
        ),
        pytest.param(
            f"Which film came out first, {FIRST_FILM} or {SECOND_FILM}?",
            [SECOND_FILM, FIRST_FILM],
            [f"Film {i}" for i in range(15)],
            [FIRST_FILM, SECOND_FILM],
            id="both-comparison-titles",
        ),
        pytest.param(
            "Which film did the director Howard Hawks make?",
            ["Film", "Director", HOWARD],
            [],
            [HOWARD],
            id="lowercase-template-words",
        ),
        pytest.param(
            "Why did Jane Pierce's husband die?",
            ["Why", JANE],
            [],
            [JANE],
            id="first-word-singleton",
        ),
        pytest.param(
            "Did Howard Hawks make this?",
            ["Howard", HOWARD],
            [],
            [HOWARD],
            id="outermost-name",
        ),
        pytest.param(
            "did jane pierce and howard hawks meet?",
            [JANE, HOWARD],
            [HOWARD],
            [HOWARD],
            id="lowercase-pool-only",
        ),
    ],
)
async def test_exact_name_anchors(
    question: str, stored: list[str], pool: list[str], expected: list[str]
) -> None:
    repo = create_autospec(GraphRepository, instance=True)
    repo.search_entities_text.return_value = [EntityHit(name, name, "Entity", 0.5) for name in pool]
    repo.entity_names_norm.side_effect = lambda ids: {name: fold(name) for name in ids}

    def find_names(names: Sequence[str]) -> list[tuple[str, EntityHit]]:
        return [
            (name_norm, EntityHit(name, name, "Entity", 1.0))
            for name in stored
            if (name_norm := fold(name)) in names
        ]

    repo.find_entities_by_name_norms.side_effect = find_names
    config = RetrievalConfig()

    hits = await EntityLinker(repo, config).link(question, [])

    assert [(hit.name, hit.score) for hit in hits] == [(name, 1.0) for name in expected]
    repo.search_entities_text.assert_awaited_once_with(fulltext_query(question), config.link_k * 3)
    repo.search_entities_vector.assert_not_awaited()
    if question.islower():
        repo.find_entities_by_name_norms.assert_not_awaited()
