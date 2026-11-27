"""Serve Ontology Studio and run its library calls in local background jobs."""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import socket
import stat
import tempfile
import time
import webbrowser
from collections.abc import AsyncIterator, Awaitable, Callable, Collection
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from threading import Thread
from typing import Literal
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, ValidationError
from pydantic_settings import SettingsError

from rag_with_oag.config import LLMConfig, resolve_api_key
from rag_with_oag.errors import OntologyError
from rag_with_oag.ingest.loaders import iter_paths
from rag_with_oag.llm.client import ChatModel, LLMClient
from rag_with_oag.ontology.model import Ontology
from rag_with_oag.ontology.questions import load_questions
from rag_with_oag.ontology.suggest import load_documents, suggest_with_coverage
from rag_with_oag.types import Document

logger = logging.getLogger(__name__)
STATIC = Path(__file__).with_name("static")
DEFAULTS = {"hops": 3, "documents": 5, "chunks_per_doc": 6}
MAX_JOBS = 50
MAX_RUNNING_JOBS = 4
PAGE_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
        "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
    ),
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}
JobOperation = Callable[[ChatModel], Awaitable[dict[str, object]]]


class InputError(ValueError):
    def __init__(self, *errors: str) -> None:
        self.errors = [_display_text(error) for error in errors]
        super().__init__("\n".join(self.errors))


def _display_text(text: str) -> str:
    return text.encode("utf-8", errors="backslashreplace").decode("utf-8")


def _validation_errors(exc: ValidationError | RequestValidationError) -> list[str]:
    return [
        _display_text(f"{'.'.join(map(str, item['loc'])) or 'ontology'}: {item['msg']}")
        for item in exc.errors()
    ]


def _path_label(root: Path, path: Path) -> str:
    try:
        return os.path.relpath(root / path, root)
    except ValueError:  # A different drive on Windows.
        return path.name


def _path_inside(root: Path, path: Path) -> Path:
    try:
        resolved = (root / path.expanduser()).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise InputError(f"cannot resolve path: {_path_label(root, path)}") from exc
    if not resolved.is_relative_to(root):
        raise InputError(f"path must be inside working directory: {_path_label(root, path)}")
    return resolved


def _parse_yaml(text: str) -> Ontology:
    try:
        # A bare filename is accepted by from_yaml. Force it to treat API input as text.
        if "\n" not in text and text.endswith((".yaml", ".yml")):
            text = "\n" + text
        ontology = Ontology.from_yaml(text)
        # YAML escapes can represent lone surrogates, which cannot travel in API responses.
        ontology.model_dump_json()
        return ontology
    except ValidationError as exc:
        raise InputError(*_validation_errors(exc)) from exc
    except (
        OntologyError,
        AttributeError,
        ValueError,
        TypeError,
        OverflowError,
        RecursionError,
    ) as exc:
        raise InputError(
            *(line.strip().removeprefix("- ") for line in str(exc).splitlines())
        ) from exc


def _write_new_file(path: Path, text: str) -> None:
    handle = path.open("x", encoding="utf-8")  # Exclusive creation with mode 0o666 & ~umask.
    try:
        with handle:
            handle.write(text)
    except OSError:
        with suppress(OSError):
            path.unlink()
        raise


def _save_file(path: Path, text: str, *, overwrite: bool) -> None:
    with tempfile.TemporaryDirectory(dir=path.parent) as staging:
        staged_file = Path(staging) / "ontology"
        _write_new_file(staged_file, text)
        if overwrite:
            with suppress(FileNotFoundError):
                staged_file.chmod(stat.S_IMODE(path.stat().st_mode))
            staged_file.replace(path)
        else:
            try:
                os.link(staged_file, path)
            except FileExistsError:
                raise
            except OSError:
                _write_new_file(path, text)


async def _run_in_worker[T](operation: Callable[[], Awaitable[T]]) -> T:
    """Give library coroutines with synchronous preprocessing their own event loop."""
    server_loop = asyncio.get_running_loop()
    started: asyncio.Future[tuple[asyncio.AbstractEventLoop, asyncio.Task[T]]] = (
        server_loop.create_future()
    )

    async def run() -> T:
        task = asyncio.current_task()
        assert task is not None
        server_loop.call_soon_threadsafe(started.set_result, (asyncio.get_running_loop(), task))
        return await operation()

    worker = asyncio.create_task(asyncio.to_thread(lambda: asyncio.run(run())))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        loop, task = await started
        with suppress(RuntimeError):  # The worker may already have completed.
            loop.call_soon_threadsafe(task.cancel)
        await asyncio.gather(worker, return_exceptions=True)
        raise


class _Body(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class _Yaml(_Body):
    yaml: str


class _Render(_Body):
    ontology: object


class _Save(_Yaml):
    path: str | None = None
    overwrite: bool = False


class _Suggest(_Body):
    questions: list[str] = Field(default_factory=list)
    instructions: str | None = None
    base_yaml: str | None = None
    repair: bool = True
    hops: PositiveInt = DEFAULTS["hops"]


class _Preview(_Yaml):
    documents: PositiveInt = DEFAULTS["documents"]
    chunks_per_doc: PositiveInt = DEFAULTS["chunks_per_doc"]


class _Check(_Yaml):
    questions: list[str]
    hops: PositiveInt = DEFAULTS["hops"]


@dataclass(slots=True)
class _Job:
    id: str
    kind: str
    status: Literal["running", "done", "error"] = "running"
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    result: dict[str, object] | None = None
    error: str | None = None
    started: float = field(default_factory=time.monotonic)
    elapsed: float | None = None

    def snapshot(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": self.elapsed
            if self.elapsed is not None
            else time.monotonic() - self.started,
            "result": self.result,
            "error": self.error,
        }


def _default_config() -> LLMConfig:
    try:
        return LLMConfig()  # type: ignore[call-arg]  # filled from env
    except ValidationError as exc:
        missing = [f"RAG_OAG_LLM_{str(item['loc'][0]).upper()}" for item in exc.errors()]
        if exc.errors()[0]["type"] == "missing":
            raise InputError(f"missing {', '.join(missing)}; configure the chat model") from exc
        raise InputError(f"invalid model settings: {', '.join(missing)}") from exc
    except SettingsError as exc:
        raise InputError("invalid RAG_OAG_LLM settings") from exc


async def _close_model(llm: ChatModel) -> None:
    close = getattr(llm, "aclose", None)
    if close is not None:
        await close()


def _public_error(message: str, root: Path, llm: ChatModel | None = None) -> str:
    keys = [os.getenv("RAG_OAG_LLM_API_KEY"), os.getenv("OPENAI_API_KEY")]
    if isinstance(llm, LLMClient):
        keys.append(resolve_api_key(llm.config.api_key, llm.config.base_url))
    for key in sorted({key for key in keys if key and key != "not-needed"}, key=len, reverse=True):
        message = message.replace(key, "[redacted]")
    return _display_text(message.replace(f"{root}{os.sep}", ""))


def create_app(
    *,
    root: Path,
    sources: list[Path],
    ontology_path: Path | None,
    questions_path: Path | None,
    token: str,
    llm_factory: Callable[[], ChatModel] | None = None,
    port: int = 8765,
    allowed_hosts: Collection[str] | None = None,
) -> FastAPI:
    """Build a token-protected studio for one working directory."""
    root = root.resolve()
    sources = [_path_inside(root, path) for path in sources]
    for path in sources:
        if not path.exists():
            raise InputError(f"no such file or directory: {_path_label(root, path)}")
    current_ontology_path = _path_inside(root, ontology_path) if ontology_path is not None else None
    if questions_path is not None:
        questions_path = _path_inside(root, questions_path)

    hosts = (
        set(allowed_hosts)
        if allowed_hosts is not None
        else {f"127.0.0.1:{port}", f"localhost:{port}"}
    )
    origins = {f"http://{host}" for host in hosts}
    token_bytes = token.encode()
    jobs: dict[str, _Job] = {}
    tasks: set[asyncio.Task[None]] = set()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            pending = list(tasks)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def protect(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.headers.get("host") not in hosts:
            return JSONResponse({"detail": "invalid Host header"}, status_code=400)
        is_api = request.url.path.startswith("/api/")
        if is_api and not secrets.compare_digest(
            request.headers.get("x-studio-token", "").encode(), token_bytes
        ):
            return JSONResponse({"detail": "invalid or missing studio token"}, status_code=401)
        origin = request.headers.get("origin")
        if request.method == "POST" and origin is not None and origin not in origins:
            return JSONResponse({"detail": "invalid Origin header"}, status_code=403)
        response = await call_next(request)
        if is_api:
            response.headers["Cache-Control"] = "no-store"
        else:
            response.headers.update(PAGE_HEADERS)
        return response

    @app.exception_handler(InputError)
    async def input_error(_request: Request, exc: InputError) -> JSONResponse:
        return JSONResponse({"errors": exc.errors}, status_code=400)

    @app.exception_handler(RequestValidationError)
    async def request_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse({"errors": _validation_errors(exc)}, status_code=400)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def document_paths() -> tuple[list[Path], int]:
        paths = []
        skipped = 0
        for source in sources:
            try:
                checked_source = _path_inside(root, source)
            except InputError as exc:
                logger.warning("Skipping Studio source: %s", exc)
                skipped += 1
                continue
            try:
                for path in iter_paths([checked_source]):
                    try:
                        paths.append(_path_inside(root, path))
                    except InputError as exc:
                        logger.warning("Skipping Studio document: %s", exc)
                        skipped += 1
            except OSError as exc:
                label = _path_label(root, source)
                reason = exc.strerror or "source is unavailable"
                raise InputError(f"could not list {label}: {reason}") from exc
        return paths, skipped

    def load_studio_documents() -> list[Document]:
        paths, _ = document_paths()
        return [
            replace(doc, source=_path_label(root, Path(doc.source)))
            for doc in load_documents(paths)
        ]

    def read_workspace(requested: Path | None) -> dict[str, object]:
        paths, skipped = document_paths()
        path = _path_inside(root, requested) if requested is not None else None
        ontology_yaml = None
        questions = []
        if path is not None:
            try:
                ontology_yaml = path.read_text(encoding="utf-8") if path.exists() else None
            except (OSError, UnicodeError) as exc:
                label = _path_label(root, path)
                raise InputError(f"could not read ontology {label}") from exc
        if questions_path is not None:
            question_file = _path_inside(root, questions_path)
            label = _path_label(root, question_file)
            try:
                questions = load_questions(question_file)
            except OSError as exc:
                raise InputError(f"could not read questions {label}") from exc
            except ValueError as exc:
                raise InputError(str(exc).replace(str(question_file), label)) from exc
        return {
            "root": str(root),
            "documents": [_path_label(root, path) for path in paths],
            "document_count": len(paths),
            "skipped_documents": skipped,
            "ontology_path": _path_label(root, path) if path is not None else None,
            "ontology_yaml": ontology_yaml,
            "questions": questions,
            "defaults": DEFAULTS,
        }

    def make_model() -> ChatModel:
        return llm_factory() if llm_factory is not None else LLMClient(_default_config())

    async def model_status() -> tuple[str | None, str | None]:
        llm = None
        try:
            if llm_factory is None:
                return (await asyncio.to_thread(_default_config)).model, None
            llm = await asyncio.to_thread(llm_factory)
            try:
                return llm.model, None
            finally:
                await _close_model(llm)
        except Exception as exc:
            return None, _public_error(str(exc), root, llm)

    @app.get("/api/state")
    async def state() -> dict[str, object]:
        data = await asyncio.to_thread(read_workspace, current_ontology_path)
        model, model_error = await model_status()
        return data | {"model": model, "model_error": model_error}

    @app.post("/api/ontology/parse")
    def parse(body: _Yaml) -> dict[str, object]:
        try:
            return {"ok": True, "ontology": _parse_yaml(body.yaml).model_dump(mode="json")}
        except InputError as exc:
            return {"ok": False, "errors": exc.errors}

    @app.post("/api/ontology/render")
    def render(body: _Render) -> dict[str, object]:
        try:
            ontology = Ontology.model_validate(body.ontology)
            ontology.model_dump_json()
            return {"ok": True, "yaml": ontology.to_yaml()}
        except ValidationError as exc:
            return {"ok": False, "errors": _validation_errors(exc)}
        except (OntologyError, ValueError, RecursionError) as exc:
            errors = [_display_text(line.removeprefix("- ")) for line in str(exc).splitlines()]
            return {"ok": False, "errors": errors}

    @app.post("/api/ontology/save", response_model=None)
    def save(body: _Save) -> dict[str, str] | JSONResponse:
        nonlocal current_ontology_path
        _parse_yaml(body.yaml)
        requested = (
            Path(body.path)
            if body.path is not None
            else current_ontology_path or Path("ontology.yaml")
        )
        path = _path_inside(root, requested)
        if any(p.suffix.lower() not in {".yaml", ".yml"} for p in (requested, path)):
            raise InputError("ontology path must end in .yaml or .yml")
        try:
            _save_file(path, body.yaml, overwrite=body.overwrite)
        except FileExistsError:
            return JSONResponse({"exists": True}, status_code=409)
        except OSError as exc:
            label = _path_label(root, requested)
            raise InputError(
                f"could not save {label}: {exc.strerror or 'file operation failed'}"
            ) from exc
        current_ontology_path = path
        return {"path": _path_label(root, path)}

    async def execute(job: _Job, llm: ChatModel, operation: JobOperation) -> None:
        async def invoke() -> dict[str, object]:
            try:
                return await operation(llm)
            finally:
                await _close_model(llm)

        try:
            job.result = await _run_in_worker(invoke)
            job.status = "done"
        except Exception as exc:
            logger.exception("Studio %s job %s failed", job.kind, job.id)
            message = _public_error(str(exc), root, llm)
            job.error = f"{type(exc).__name__}: {message}"
            job.status = "error"
        finally:
            job.finished_at = datetime.now(UTC)
            job.elapsed = time.monotonic() - job.started

    async def start_job(kind: str, operation: JobOperation) -> dict[str, str]:
        if sum(job.status == "running" for job in jobs.values()) >= MAX_RUNNING_JOBS:
            raise HTTPException(
                status_code=429,
                detail=f"All {MAX_RUNNING_JOBS} job slots are busy. Wait for a job to finish.",
            )
        job = _Job(uuid4().hex, kind)
        jobs[job.id] = job
        try:
            llm = await asyncio.to_thread(make_model)
        except asyncio.CancelledError:
            del jobs[job.id]
            raise
        except Exception as exc:
            del jobs[job.id]
            raise InputError(_public_error(str(exc), root)) from exc
        while len(jobs) > MAX_JOBS:
            oldest_finished = next(key for key, item in jobs.items() if item.status != "running")
            del jobs[oldest_finished]
        task = asyncio.create_task(execute(job, llm, operation))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return {"job_id": job.id}

    @app.post("/api/jobs/suggest")
    async def suggest(body: _Suggest) -> dict[str, str]:
        base = (
            await asyncio.to_thread(_parse_yaml, body.base_yaml)
            if body.base_yaml is not None
            else None
        )

        async def operation(llm: ChatModel) -> dict[str, object]:
            ontology, coverage = await suggest_with_coverage(
                load_studio_documents(),
                llm,
                base=base,
                questions=body.questions,
                instructions=body.instructions,
                repair=body.repair,
                hops=body.hops,
            )
            return {
                "yaml": ontology.to_yaml(),
                "ontology": ontology.model_dump(mode="json"),
                "coverage": asdict(coverage) if coverage is not None else None,
            }

        return await start_job("suggest", operation)

    @app.post("/api/jobs/preview")
    async def preview(body: _Preview) -> dict[str, str]:
        ontology = await asyncio.to_thread(_parse_yaml, body.yaml)

        async def operation(llm: ChatModel) -> dict[str, object]:
            report = await ontology.apreview(
                load_studio_documents(),
                llm,
                documents=body.documents,
                chunks_per_doc=body.chunks_per_doc,
            )
            report.failed_batches = [
                _public_error(failure, root, llm) for failure in report.failed_batches
            ]
            return asdict(report)

        return await start_job("preview", operation)

    @app.post("/api/jobs/check-questions")
    async def check_questions(body: _Check) -> dict[str, str]:
        ontology = await asyncio.to_thread(_parse_yaml, body.yaml)
        if not any(question.strip() for question in body.questions):
            raise InputError("coverage check needs at least one question")

        async def operation(llm: ChatModel) -> dict[str, object]:
            return asdict(await ontology.acheck_questions(body.questions, llm, hops=body.hops))

        return await start_job("check-questions", operation)

    @app.get("/api/jobs/{job_id}")
    async def job_state(job_id: str) -> dict[str, object]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return job.snapshot()

    return app


def run(app: FastAPI, *, port: int, open_browser: bool, url: str) -> None:
    """Bind loopback before launching the browser, and serve until interrupted."""
    host = "127.0.0.1"
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, proxy_headers=False))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        if os.name != "nt":
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen()
        if open_browser:
            Thread(target=webbrowser.open, args=(url,), daemon=True).start()
        server.run(sockets=[listener])
