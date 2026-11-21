from __future__ import annotations

import sys
import threading
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rag_with_oag.rerank.local import FastEmbedReranker


async def test_fastembed_reranker_collects_scores_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_loop_thread = threading.get_ident()

    def scores(*args: object, **kwargs: object) -> Iterator[float]:
        assert threading.get_ident() != event_loop_thread
        yield from (0.25, 0.75)

    rerank = Mock(side_effect=scores)
    constructor = Mock(return_value=SimpleNamespace(rerank=rerank))
    monkeypatch.setitem(
        sys.modules, "fastembed.rerank.cross_encoder", SimpleNamespace(TextCrossEncoder=constructor)
    )

    model = FastEmbedReranker(batch_size=2)
    texts = ("first", "second")

    constructor.assert_called_once_with(model_name="jinaai/jina-reranker-v2-base-multilingual")
    assert await model.rerank("question", texts) == [0.25, 0.75]
    rerank.assert_called_once_with("question", texts, batch_size=2)
