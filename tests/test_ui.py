"""Exercise Studio with in-process HTTP and queued model replies."""

from __future__ import annotations

import asyncio
import builtins
import errno
import json
import os
import stat
import subprocess
import sys
import time
from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from typing import Any

import pytest

from rag_with_oag import Ontology
from rag_with_oag.cli import main
from rag_with_oag.llm.client import Message
from rag_with_oag.types import Document

from .conftest import FakeLLM

pytest.importorskip("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag_with_oag import ui
from rag_with_oag.ui import create_app, server

TOKEN = "studio-token"
HEADERS = {"X-Studio-Token": TOKEN}
YAML = """\
name: contracts
entities:
  Company:
    key: tax_number
    properties: {tax_number: string}
  Contract: {}
relations:
  SIGNS: {source: Company, target: Contract}
"""
QUESTION = "Which company signed this contract?"
PATH_REPLY = {
    "answers": [
        {
            "index": 1,
            "route": "graph",
            "route_reason": "",
            "answerable": True,
            "start_type": "Contract",
            "steps": [{"relation": "SIGNS", "direction": "backward"}],
            "answer": {"type": "Company", "property": "name"},
            "missing": "",
        }
    ]
}
PREVIEW_REPLY = {
    "entities": [
        {
            "ref": "company",
            "name": "Acme",
            "type": "Company",
            "chunks": ["K1"],
            "properties": {"tax_number": "123"},
        },
        {"ref": "contract", "name": "C-1", "type": "Contract", "chunks": ["K1"]},
    ],
    "relations": [{"source": "company", "target": "contract", "type": "SIGNS"}],
    "unmapped": [],
}


class QueuedLLM(FakeLLM):
    def __init__(self, *replies: dict[str, Any] | Exception) -> None:
        super().__init__()
        self.replies = deque(replies)
        self.closed = False

    async def complete(self, messages: Sequence[Message], **kwargs: Any) -> str:
        reply = self.replies.popleft()
        if isinstance(reply, Exception):
            raise reply
        self.reply = json.dumps(reply)
        return await super().complete(messages, **kwargs)

    async def aclose(self) -> None:
        self.closed = True


class WaitingLLM(QueuedLLM):
    def __init__(self, release: Event) -> None:
        super().__init__(PATH_REPLY)
        self.release = release

    async def complete(self, messages: Sequence[Message], **kwargs: Any) -> str:
        await asyncio.to_thread(self.release.wait)
        return await super().complete(messages, **kwargs)


@pytest.fixture(autouse=True)
def offline_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_download(*args: Any, **kwargs: Any) -> bytes:
        raise OSError("tokenizer downloads are disabled in this test")

    monkeypatch.setattr("tiktoken.load.read_file", no_download)


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    docs = root / "docs"
    docs.mkdir(parents=True)
    (docs / "contract.txt").write_text("Acme (tax number 123) signed contract C-1.")
    (docs / "terms.md").write_text("# Contract terms\nAcme must pay the invoice.")
    (docs / ".hidden.txt").write_text("This should be skipped.")
    (docs / "data.bin").write_bytes(b"\x00")
    return root


def _client(root: Path, **options: Any) -> TestClient:
    kwargs = {"ontology_path": None, "questions_path": None} | options
    app = create_app(
        root=root, sources=[Path("docs")], token=TOKEN, allowed_hosts={"testserver"}, **kwargs
    )
    return TestClient(app, headers=HEADERS)


@pytest.fixture
def client(workspace: Path) -> Iterator[TestClient]:
    with _client(workspace, llm_factory=QueuedLLM) as client:
        yield client


def test_token_host_origin_and_static(client: TestClient) -> None:
    for token in ("", "wrong"):
        assert client.get("/api/state", headers={"X-Studio-Token": token}).status_code == 401
    with TestClient(client.app, base_url="http://testserver") as anonymous:
        assert anonymous.get("/api/state").status_code == 401
        for path in ("/", "/static/app.js", "/static/app.css", "/static/vendor/cytoscape.min.js"):
            response = anonymous.get(path)
            assert response.status_code == 200
            assert response.headers["Content-Security-Policy"] == (
                "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
            )
            assert response.headers["X-Frame-Options"] == "DENY"
            assert response.headers["X-Content-Type-Options"] == "nosniff"
            assert response.headers["Referrer-Policy"] == "no-referrer"
        assert "Ontology Studio" in anonymous.get("/").text
        script = anonymous.get("/static/app.js").text
        assert "confirm(" not in script
        assert "alert(" not in script

    assert client.get("/api/state", headers={"Host": "evil.example"}).status_code == 400
    assert client.get("/", headers={"Host": "evil.example"}).status_code == 400
    for origin in ("http://evil.example", "https://testserver", "null"):
        result = client.post("/api/ontology/parse", json={"yaml": YAML}, headers={"Origin": origin})
        assert result.status_code == 403
    result = client.post(
        "/api/ontology/parse", json={"yaml": YAML}, headers={"Origin": "http://testserver"}
    )
    assert result.json()["ok"] is True
    assert client.get("/api/state").status_code == 200


def test_loopback_default_hosts_use_port(workspace: Path) -> None:
    app = create_app(
        root=workspace,
        sources=[Path("docs")],
        ontology_path=None,
        questions_path=None,
        token=TOKEN,
        port=9321,
    )
    with TestClient(app, base_url="http://127.0.0.1:9321", headers=HEADERS) as client:
        assert client.get("/api/state").status_code == 200
        assert client.get("/api/state", headers={"Host": "localhost:9321"}).status_code == 200
        assert client.get("/api/state", headers={"Host": "localhost:8765"}).status_code == 400


def test_state(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_OAG_LLM_MODEL", raising=False)
    (workspace / "existing.yml").write_text(YAML)
    (workspace / "questions.txt").write_text(f"# Ignore this\n\n{QUESTION}\n")
    with _client(
        workspace, ontology_path=Path("existing.yml"), questions_path=Path("questions.txt")
    ) as client:
        state = client.get("/api/state").json()
        assert state == {
            "root": str(workspace.resolve()),
            "documents": ["docs/contract.txt", "docs/terms.md"],
            "document_count": 2,
            "skipped_documents": 0,
            "ontology_path": "existing.yml",
            "ontology_yaml": YAML,
            "questions": [QUESTION],
            "model": None,
            "model_error": "missing RAG_OAG_LLM_MODEL; configure the chat model",
            "defaults": {"hops": 3, "documents": 5, "chunks_per_doc": 6},
        }

    monkeypatch.setenv("RAG_OAG_LLM_MODEL", "local-model")
    monkeypatch.setenv("RAG_OAG_LLM_API_KEY", "private-api-key")
    with _client(workspace, ontology_path=Path("new.yml")) as client:
        response = client.get("/api/state")
        assert response.json()["ontology_path"] == "new.yml"
        assert response.json()["ontology_yaml"] is None
        assert response.json()["model"] == "local-model"
        assert response.json()["model_error"] is None
        assert "private-api-key" not in response.text


@pytest.mark.parametrize(
    "yaml",
    [
        "entities: [",
        "entities: {}",
        "entities: {Company: yes}",
        "file.yml",
        'name: "\\ud800"\nentities: {Company: {}}',
    ],
)
def test_parse_rejects_bad_yaml(client: TestClient, yaml: str) -> None:
    response = client.post("/api/ontology/parse", json={"yaml": yaml})
    assert response.status_code == 200
    assert response.json()["ok"] is False
    assert response.json()["errors"]


def test_parse_uses_text_even_when_it_looks_like_a_path(
    client: TestClient, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(workspace)
    (workspace / "source.yaml").write_text(YAML)
    result = client.post("/api/ontology/parse", json={"yaml": "source.yaml"}).json()
    assert result["ok"] is False


def test_round_trip_and_render_errors(client: TestClient) -> None:
    parsed = client.post("/api/ontology/parse", json={"yaml": YAML}).json()
    schema = Ontology.from_yaml(YAML)
    assert parsed == {"ok": True, "ontology": schema.model_dump(mode="json")}
    rendered = client.post("/api/ontology/render", json={"ontology": parsed["ontology"]}).json()
    assert rendered["ok"]
    assert Ontology.from_yaml(rendered["yaml"]) == schema

    for ontology in ({"entities": []}, {"entities": [{"name": []}]}, None):
        response = client.post("/api/ontology/render", json={"ontology": ontology})
        assert response.status_code == 200
        assert response.json()["ok"] is False
        assert response.json()["errors"]


def test_save_and_overwrite(client: TestClient, workspace: Path) -> None:
    base = {"yaml": YAML, "path": None, "overwrite": False}
    route = "/api/ontology/save"
    assert client.post(route, json=base).json() == {"path": "ontology.yaml"}
    assert Ontology.from_yaml(workspace / "ontology.yaml") == Ontology.from_yaml(YAML)
    state = client.get("/api/state").json()
    assert state["ontology_path"] == "ontology.yaml"
    assert state["ontology_yaml"] == YAML
    assert client.post(route, json=base | {"path": "docs/other.yml"}).json() == {
        "path": "docs/other.yml"
    }
    target = workspace / "docs" / "other.yml"
    assert client.get("/api/state").json()["ontology_path"] == "docs/other.yml"
    conflict = client.post(route, json=base)
    assert conflict.status_code == 409
    assert conflict.json() == {"exists": True}
    updated = YAML.replace("name: contracts", "name: updated")
    assert client.post(route, json=base | {"yaml": updated, "overwrite": True}).status_code == 200
    assert Ontology.from_yaml(target).name == "updated"
    state = client.get("/api/state").json()
    assert state["ontology_path"] == "docs/other.yml"
    assert state["ontology_yaml"] == updated
    bad = client.post(route, json=base | {"yaml": "entities: [", "overwrite": True})
    assert bad.status_code == 400
    assert bad.json()["errors"]
    assert target.read_text() == updated


def test_save_respects_configured_path(workspace: Path) -> None:
    with _client(workspace, ontology_path=Path("planned.yml")) as client:
        assert client.post("/api/ontology/save", json={"yaml": YAML}).json() == {
            "path": "planned.yml"
        }
    assert Ontology.from_yaml(workspace / "planned.yml") == Ontology.from_yaml(YAML)


@pytest.mark.skipif(os.name == "nt", reason="needs POSIX file modes")
def test_save_modes(client: TestClient, workspace: Path) -> None:
    route = "/api/ontology/save"
    target = workspace / "ontology.yaml"
    mask = os.umask(0o027)
    try:
        assert client.post(route, json={"yaml": YAML}).status_code == 200
        assert stat.S_IMODE(target.stat().st_mode) == 0o640
        target.chmod(0o664)
        assert client.post(route, json={"yaml": YAML, "overwrite": True}).status_code == 200
        assert stat.S_IMODE(target.stat().st_mode) == 0o664
    finally:
        os.umask(mask)


def test_save_falls_back_to_exclusive_create(
    client: TestClient, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unsupported_link(*args: Any) -> None:
        raise OSError(errno.EOPNOTSUPP, "hard links unavailable")

    monkeypatch.setattr(server.os, "link", unsupported_link)
    route = "/api/ontology/save"
    assert client.post(route, json={"yaml": YAML}).status_code == 200
    updated = YAML.replace("name: contracts", "name: updated")
    assert client.post(route, json={"yaml": updated}).status_code == 409
    assert (workspace / "ontology.yaml").read_text() == YAML


def test_save_error_reports_only_requested_path(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_save(path: Path, text: str, *, overwrite: bool) -> None:
        raise PermissionError(errno.EACCES, "Permission denied", path.parent / "tmpx22q1qph")

    monkeypatch.setattr(server, "_save_file", failing_save)
    response = client.post("/api/ontology/save", json={"yaml": YAML, "path": "docs/new.yml"})
    assert response.status_code == 400
    assert response.json() == {"errors": ["could not save docs/new.yml: Permission denied"]}
    assert client.get("/api/state").json()["ontology_path"] is None


@pytest.mark.parametrize("path", ["../outside.yaml", "notes.txt", "escape.yaml"])
def test_save_rejects_invalid_paths(client: TestClient, workspace: Path, path: str) -> None:
    outside = workspace.parent / "outside.yaml"
    (workspace / "escape.yaml").symlink_to(outside)
    response = client.post(
        "/api/ontology/save", json={"yaml": YAML, "path": path, "overwrite": True}
    )
    assert response.status_code == 400
    assert response.json()["errors"]
    assert not outside.exists()


def test_skips_escaping_document(workspace: Path, caplog: pytest.LogCaptureFixture) -> None:
    outside = workspace.parent / "outside.txt"
    outside.write_text("Private document outside the workspace.")
    (workspace / "docs" / "escape.txt").symlink_to(outside)
    llm = QueuedLLM(PREVIEW_REPLY, PREVIEW_REPLY)
    models = iter((QueuedLLM(), llm))
    with _client(workspace, llm_factory=lambda: next(models)) as client:
        state = client.get("/api/state")
        assert state.status_code == 200
        assert state.json()["documents"] == ["docs/contract.txt", "docs/terms.md"]
        assert state.json()["skipped_documents"] == 1
        assert "Skipping Studio document" in caplog.text
        job = _finish(client, "preview", {"yaml": YAML, "documents": 2})
        assert job["status"] == "done"
        assert job["result"]["documents_available"] == 2
    assert "Private document outside" not in str(llm.requests)


def _finish(client: TestClient, kind: str, body: dict[str, Any]) -> dict[str, Any]:
    response = client.post(f"/api/jobs/{kind}", json=body)
    assert response.status_code == 200, response.text
    job_id = response.json()["job_id"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()
        assert job["id"] == job_id
        assert job["kind"] == kind
        assert job["started_at"]
        assert job["elapsed_seconds"] >= 0
        if job["status"] != "running":
            assert job["finished_at"]
            return job
        time.sleep(0.01)
    pytest.fail(f"{kind} job did not finish")


def test_running_job_survives_eviction(workspace: Path) -> None:
    release = Event()
    models = deque([WaitingLLM(release)])

    def make_model() -> QueuedLLM:
        return models.popleft() if models else QueuedLLM(PATH_REPLY)

    body = {"yaml": YAML, "questions": [QUESTION]}
    with _client(workspace, llm_factory=make_model) as client:
        try:
            running = client.post("/api/jobs/check-questions", json=body).json()["job_id"]
            finished = [_finish(client, "check-questions", body)["id"] for _ in range(50)]
            assert client.get(f"/api/jobs/{running}").json()["status"] == "running"
            assert client.get(f"/api/jobs/{finished[0]}").status_code == 404
            assert client.get(f"/api/jobs/{finished[-1]}").json()["status"] == "done"
        finally:
            release.set()


def test_jobs_return_429_at_running_limit(workspace: Path) -> None:
    release = Event()
    body = {"yaml": YAML, "questions": [QUESTION]}
    route = "/api/jobs/check-questions"
    with _client(workspace, llm_factory=lambda: WaitingLLM(release)) as client:
        try:
            for _ in range(4):
                assert client.post(route, json=body).status_code == 200
            response = client.post(route, json=body)
            assert response.status_code == 429
            assert "All 4 job slots are busy" in response.json()["detail"]
        finally:
            release.set()


def test_shutdown_cancels_worker_job_and_closes_model(workspace: Path) -> None:
    started = Event()
    cancelled = Event()

    class PendingLLM(QueuedLLM):
        async def complete(self, messages: Sequence[Message], **kwargs: Any) -> str:
            started.set()
            try:
                await asyncio.wait_for(asyncio.Event().wait(), timeout=5)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return await super().complete(messages, **kwargs)

    model = PendingLLM()
    with _client(workspace, llm_factory=lambda: model) as client:
        body = {"yaml": YAML, "questions": [QUESTION]}
        assert client.post("/api/jobs/check-questions", json=body).status_code == 200
        assert started.wait(timeout=2)
    assert cancelled.is_set()
    assert model.closed


def test_state_walks_files_in_a_worker(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    iter_paths = server.iter_paths

    def walk(sources: list[Path]) -> Iterator[Path]:
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        return iter_paths(sources)

    monkeypatch.setattr(server, "iter_paths", walk)
    assert client.get("/api/state").status_code == 200


def test_preview_preparation_does_not_block_polling(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = Event()
    release = Event()
    apreview = Ontology.apreview

    async def prepare(
        self: Ontology, sources: list[Document], llm: QueuedLLM, **kwargs: Any
    ) -> Any:
        assert [doc.source for doc in sources] == ["docs/contract.txt", "docs/terms.md"]
        started.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release preview preparation")
        return await apreview(self, sources, llm, **kwargs)

    monkeypatch.setattr(Ontology, "apreview", prepare)
    with _client(workspace, llm_factory=lambda: QueuedLLM(PREVIEW_REPLY)) as client:
        response = client.post("/api/jobs/preview", json={"yaml": YAML, "documents": 1})
        assert response.status_code == 200
        job_url = f"/api/jobs/{response.json()['job_id']}"
        with ThreadPoolExecutor(max_workers=1) as thread:
            try:
                assert started.wait(timeout=2)
                result = thread.submit(client.get, job_url).result(timeout=2)
                assert result.json()["status"] == "running"
            finally:
                release.set()


def test_jobs_use_and_close_separate_models(workspace: Path) -> None:
    schema = Ontology.from_yaml(YAML)
    suggestion = schema.model_dump(mode="json", exclude={"name", "version"})
    suggest_llm = QueuedLLM(suggestion, PATH_REPLY)
    preview_llm = QueuedLLM(PREVIEW_REPLY)
    questions_llm = QueuedLLM(PATH_REPLY)
    models = iter((suggest_llm, preview_llm, questions_llm))

    with _client(workspace, llm_factory=lambda: next(models)) as client:
        suggested = _finish(client, "suggest", {"questions": [QUESTION], "base_yaml": YAML})
        assert suggested["status"] == "done", suggested
        assert suggested["error"] is None
        assert suggested["result"]["ontology"] == schema.model_dump(mode="json")
        assert Ontology.from_yaml(suggested["result"]["yaml"]) == schema
        assert suggested["result"]["coverage"]["questions"][0]["status"] == "covered"

        previewed = _finish(client, "preview", {"yaml": YAML, "documents": 1, "chunks_per_doc": 2})
        assert previewed["status"] == "done", previewed
        report = previewed["result"]
        assert report["documents_sampled"] == 1
        assert report["chunks_sampled"] == 1
        assert report["entity_types"][0]["key_fill_rate"] == 1.0
        assert report["relation_types"][0]["count"] == 1

        checked = _finish(client, "check-questions", {"yaml": YAML, "questions": [QUESTION]})
        assert checked["status"] == "done", checked
        item = checked["result"]["questions"][0]
        assert item["status"] == "covered"
        assert item["start_type"] == "Contract"
        assert item["steps"] == [["SIGNS", "backward"]]

    assert all(llm.closed for llm in (suggest_llm, preview_llm, questions_llm))


def test_job_errors_redact_key(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_OAG_LLM_API_KEY", "private-api-key")
    llm = QueuedLLM(RuntimeError("provider failed: private-api-key"))
    with _client(workspace, llm_factory=lambda: llm) as client:
        job = _finish(client, "suggest", {})
    assert job["status"] == "error"
    assert job["error"] == "RuntimeError: provider failed: [redacted]"
    assert job["result"] is None
    assert llm.closed


def test_rejects_bad_job_inputs(client: TestClient) -> None:
    invalid = {
        "suggest": [{"base_yaml": "entities: ["}, {"hops": 0}],
        "preview": [{"yaml": "entities: ["}, {"yaml": YAML, "chunks_per_doc": -1}],
        "check-questions": [{"yaml": YAML, "questions": [" "]}, {"yaml": "[", "questions": ["Q"]}],
    }
    for kind, bodies in invalid.items():
        for body in bodies:
            response = client.post(f"/api/jobs/{kind}", json=body)
            assert response.status_code == 400
            assert response.json()["errors"]
    assert client.get("/api/jobs/missing").status_code == 404


def test_jobs_rejected_without_model(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAG_OAG_LLM_MODEL", raising=False)
    with _client(workspace) as client:
        for kind, body in (
            ("suggest", {}),
            ("preview", {"yaml": YAML}),
            ("check-questions", {"yaml": YAML, "questions": [QUESTION]}),
        ):
            response = client.post(f"/api/jobs/{kind}", json=body)
            assert response.status_code == 400
            assert "missing RAG_OAG_LLM_MODEL" in response.json()["errors"][0]

    def unavailable() -> QueuedLLM:
        raise ValueError("model configuration unavailable")

    with _client(workspace, llm_factory=unavailable) as client:
        state = client.get("/api/state").json()
        assert state["model"] is None
        assert state["model_error"] == "model configuration unavailable"
        assert client.post("/api/jobs/suggest", json={}).status_code == 400


@pytest.mark.parametrize("option", [None, "--ontology", "--questions"])
def test_cli_rejects_paths_outside_root(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    option: str | None,
) -> None:
    monkeypatch.chdir(workspace)
    outside = workspace.parent / "outside.yaml"
    args = [str(outside)] if option is None else ["docs", option, str(outside)]
    assert main(["ui", *args]) == 2
    assert "inside working directory" in capsys.readouterr().err


def test_cli_without_ui_extra(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real_import = builtins.__import__

    def missing_ui(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "rag_with_oag.ui":
            raise ModuleNotFoundError("No module named 'fastapi'", name="fastapi")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_ui)
    assert main(["ui", "docs"]) == 2
    assert "pip install 'rag_with_oag[ui]'" in capsys.readouterr().err


@pytest.mark.parametrize("no_browser", [False, True])
def test_cli_run(
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_browser: bool,
) -> None:
    monkeypatch.chdir(workspace)
    calls = []

    def run(app: FastAPI, *, port: int, open_browser: bool, url: str) -> None:
        calls.append((port, open_browser, url))
        token = url.split("#token=", 1)[-1]
        assert len(token) >= 24
        with TestClient(app, base_url=f"http://127.0.0.1:{port}") as client:
            assert client.get("/api/state", headers={"X-Studio-Token": token}).status_code == 200
        raise KeyboardInterrupt

    monkeypatch.setattr(ui, "run", run)
    args = ["ui", "docs", "--ontology", "new.yaml", "--port", "9321"]
    assert main(args + (["--no-browser"] if no_browser else [])) == 0
    assert len(calls) == 1
    port, open_browser, url = calls[0]
    assert port == 9321
    assert open_browser is not no_browser
    assert url.startswith("http://127.0.0.1:9321/#token=")
    output = capsys.readouterr().out
    assert f"Ontology Studio: {url}" in output
    assert "Press Ctrl+C to stop" in output


def test_cli_bind_failure(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(workspace)

    def run(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("Address already in use")

    monkeypatch.setattr(ui, "run", run)
    assert main(["ui", "docs", "--no-browser"]) == 1
    assert "cannot listen on 127.0.0.1:8765: Address already in use" in capsys.readouterr().err


def test_core_import_does_not_load_fastapi() -> None:
    subprocess.run(
        [sys.executable, "-c", "import rag_with_oag, sys; assert 'fastapi' not in sys.modules"],
        check=True,
    )
