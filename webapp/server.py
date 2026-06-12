"""Web UI for the Watts Podcast Generator.

Local dev:
    GROQ_API_KEY=... COQUI_TOS_AGREED=1 ./venv/bin/uvicorn webapp.server:app --reload

Production (Docker/Render):
    Set GROQ_API_KEY, BASIC_AUTH_USER, BASIC_AUTH_PASS, WATTS_DATA_DIR (persistent disk).
"""
import base64
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "watts_podcast.py"
DATA_DIR = Path(os.environ.get("WATTS_DATA_DIR", str(ROOT)))
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS = DATA_DIR / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
STATIC = Path(__file__).resolve().parent / "static"

BASIC_AUTH_USER = os.environ.get("BASIC_AUTH_USER")
BASIC_AUTH_PASS = os.environ.get("BASIC_AUTH_PASS")


class BasicAuthMiddleware(BaseHTTPMiddleware):
    """Gates the whole app behind HTTP Basic Auth when credentials are configured.

    Public deployments run real CPU-heavy jobs against a shared API key,
    so an open instance would be an easy abuse/cost vector.
    """

    async def dispatch(self, request, call_next):
        if not BASIC_AUTH_USER:
            return await call_next(request)

        auth = request.headers.get("authorization", "")
        if auth.startswith("Basic "):
            try:
                decoded = base64.b64decode(auth[6:]).decode()
                user, _, pw = decoded.partition(":")
                if user == BASIC_AUTH_USER and pw == BASIC_AUTH_PASS:
                    return await call_next(request)
            except Exception:
                pass

        return Response(status_code=401, headers={"WWW-Authenticate": 'Basic realm="Watts Podcast Generator"'})


app = FastAPI(title="Watts Podcast Generator")
app.add_middleware(BasicAuthMiddleware)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

jobs = {}


def _parse_result(log_text):
    for line in log_text.splitlines():
        line = line.strip()
        if line.startswith("[Done]") and line.endswith(".mp3"):
            return line.rsplit(" ", 1)[-1]
    return None


def _run_job(job_id, cmd):
    log_path = UPLOADS / f"{job_id}.log"
    env = os.environ.copy()
    env["COQUI_TOS_AGREED"] = "1"
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, cwd=str(DATA_DIR), stdout=logf, stderr=subprocess.STDOUT, env=env)
    log_text = log_path.read_text(errors="ignore")
    jobs[job_id]["status"] = "done" if proc.returncode == 0 else "error"
    jobs[job_id]["result"] = _parse_result(log_text)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/api/generate")
async def generate(
    file: UploadFile = File(...),
    mode: str = Form("new"),
    duration: int = Form(20),
    use_arc: bool = Form(True),
    use_memory: bool = Form(True),
    answer_space: float = Form(20.0),
):
    job_id = uuid.uuid4().hex[:8]
    dest = UPLOADS / f"{job_id}_{file.filename}"
    dest.write_bytes(await file.read())

    if mode == "reply":
        cmd = [sys.executable, str(SCRIPT), "--reply", str(dest)]
    else:
        cmd = [
            sys.executable, str(SCRIPT),
            "--input", str(dest),
            "--duration", str(duration),
            "--answer-space", str(answer_space),
        ]
        if not use_arc:
            cmd.append("--no-arc")
        if not use_memory:
            cmd.append("--no-memory")

    jobs[job_id] = {"status": "running", "result": None}
    threading.Thread(target=_run_job, args=(job_id, cmd), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
def status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "not found"}, status_code=404)
    log_path = UPLOADS / f"{job_id}.log"
    log_text = log_path.read_text(errors="ignore") if log_path.exists() else ""
    return {"status": job["status"], "log": log_text[-4000:], "result": job["result"]}


@app.get("/api/download/{job_id}")
def download(job_id: str):
    job = jobs.get(job_id)
    if not job or not job["result"]:
        return JSONResponse({"error": "not ready"}, status_code=404)
    path = DATA_DIR / job["result"]
    if not path.exists():
        return JSONResponse({"error": "file missing"}, status_code=404)
    return FileResponse(path, filename=path.name, media_type="audio/mpeg")


@app.get("/api/ambient")
def ambient():
    path = STATIC / "ambient.wav"
    if not path.exists():
        return JSONResponse({"error": "ambient.wav not generated yet"}, status_code=404)
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/memory")
def memory_info():
    mem_path = DATA_DIR / "watts_memory.json"
    if not mem_path.exists():
        return {"episodes": [], "threads": []}
    import json
    data = json.loads(mem_path.read_text())
    episodes = [
        {"id": e["id"], "title": e.get("title", ""), "source": e.get("source_doc", "")}
        for e in data.get("episodes", [])
    ]
    return {"episodes": episodes, "threads": data.get("listener_threads", [])}
