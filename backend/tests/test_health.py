import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

BACKEND_DIR = Path(__file__).resolve().parents[1]


def test_health_returns_200_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_cors_allows_the_frontend_origin():
    response = client.options(
        "/query",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert (
        response.headers["access-control-allow-origin"]
        == "http://localhost:3000"
    )


def test_setdefault_after_load_dotenv_lets_an_env_file_override_the_default(
    tmp_path,
):
    """Unit test, in isolation, of the ordering pattern main.py's
    `os.environ.setdefault("OMP_NUM_THREADS", "1")` (and
    `TOKENIZERS_PARALLELISM`) relies on: `load_dotenv()`'s default
    `override=False` only fills keys still absent from `os.environ`, so
    `setdefault` must run *after* `load_dotenv()`, not before -- an
    earlier version of main.py had it backwards, silently ignoring a
    value set in `.env` with no error.

    Deliberately NOT routed through `from app.main import app`:
    `app.retrieval.gemini` independently calls `load_dotenv()` too (so it
    also works standalone, e.g. in scripts) -- which means `.env` is
    already loaded as a side effect of `from app.api.routes import ...`,
    before main.py's own `load_dotenv()`/`setdefault` calls ever run,
    *regardless of their order*. Verified directly: the reversed
    (buggy) order in main.py still resolves to the `.env` value when
    routed through the full app, purely because of that unrelated
    module's side effect -- so a test going through `app.main` would
    pass or fail independent of the fix this guards, giving false
    confidence either way. This test isolates main.py's actual two-line
    pattern instead, matching it exactly, so it's meaningful on its own
    merits.

    Runs in a real subprocess: `load_dotenv()` with no argument searches
    from the current working directory, and a temp cwd (not
    `backend/.env`, the developer's real, gitignored file) keeps this
    isolated.
    """
    (tmp_path / ".env").write_text("OMP_NUM_THREADS=7\n")
    script = (
        "import os\n"
        "from dotenv import load_dotenv\n"
        "load_dotenv()\n"
        "os.environ.setdefault('OMP_NUM_THREADS', '1')\n"
        "print(os.environ.get('OMP_NUM_THREADS'))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={"PYTHONPATH": str(BACKEND_DIR), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "7"
