"""Core engine: lazy model loading, voice library, chunked generation."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
VOICES_DIR = APP_DIR / "voices"
OUTPUT_DIR = APP_DIR / "output"
MODELS_DIR = APP_DIR / "models"

TARGET_SR = 24000
MAX_CHARS = 2000
GEN_TIMEOUT = 600

LANGUAGES = [
    "English", "Chinese", "Japanese", "Korean", "German",
    "French", "Russian", "Portuguese", "Spanish", "Italian",
]

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
# The 1.7B model is far better but needs ~9GB RAM. Enable in config.json.
BIG_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

for _d in (VOICES_DIR, OUTPUT_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- config

def load_config() -> dict:
    import json
    cfg = {"model": DEFAULT_MODEL, "threads": os.cpu_count() or 4}
    path = APP_DIR / "config.json"
    if path.exists():
        try:
            cfg.update(json.loads(path.read_text()))
        except Exception:
            pass
    return cfg


CONFIG = load_config()


# ---------------------------------------------------------------- model

_model = None
_model_name: str | None = None
_model_lock = threading.Lock()


def _threads() -> int:
    t = int(CONFIG.get("threads") or os.cpu_count() or 4)
    t = max(1, min(t, os.cpu_count() or t))
    os.environ.setdefault("OMP_NUM_THREADS", str(t))
    return t


def get_model():
    """Load the model lazily, at most once. Blocks on first call."""
    global _model, _model_name
    want = CONFIG["model"]
    with _model_lock:
        if _model is not None and _model_name == want:
            return _model
        import torch
        from qwen_tts import Qwen3TTSModel

        n = _threads()
        torch.set_num_threads(n)
        print(f"[engine] loading {want} on cpu, threads={n}", flush=True)
        t0 = time.time()
        _model = Qwen3TTSModel.from_pretrained(
            want,
            device_map="cpu",
            dtype=torch.float32,
            attn_implementation="eager",  # no flash-attn on cpu
        )
        _model_name = want
        print(f"[engine] loaded in {time.time()-t0:.1f}s", flush=True)
        return _model


def model_loaded_name() -> str | None:
    return _model_name


# ---------------------------------------------------------------- audio

def _ffmpeg() -> str:
    return shutil.which("ffmpeg") or "ffmpeg"


def probe_duration(path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def to_reference_wav(src: Path, dest: Path) -> dict:
    """Normalise any audio to mono 24kHz PCM16 - what the model wants."""
    cmd = [
        _ffmpeg(), "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(TARGET_SR),
        "-c:a", "pcm_s16le", str(dest),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr.strip()[:400]}")
    return {
        "duration": probe_duration(dest),
        "size_mb": dest.stat().st_size / 1e6,
    }


def write_wav(dest: Path, samples, sr: int) -> None:
    import soundfile as sf
    sf.write(str(dest), samples, sr, subtype="PCM_16")


# ---------------------------------------------------------------- voices

@dataclass
class Voice:
    name: str
    path: Path
    duration: float = 0.0

    @property
    def size_mb(self) -> float:
        try:
            return self.path.stat().st_size / 1e6
        except OSError:
            return 0.0

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "duration": self.duration,
            "size_mb": self.size_mb,
            "path": str(self.path),
        }


def safe_name(raw: str) -> str:
    name = re.sub(r"[^A-Za-z0-9 _\-]", "", (raw or "").strip())
    name = re.sub(r"\s+", " ", name).strip(" .-_")
    return name[:60] or "voice"


def list_voices() -> list[Voice]:
    out = []
    for p in sorted(VOICES_DIR.glob("*.wav")):
        out.append(Voice(p.stem, p, probe_duration(p)))
    return out


def add_voice(name: str, src: Path) -> Voice:
    name = safe_name(name)
    if src.resolve() == (VOICES_DIR / f"{name}.wav").resolve():
        return Voice(name, VOICES_DIR / f"{name}.wav",
                     probe_duration(VOICES_DIR / f"{name}.wav"))
    dest = VOICES_DIR / f"{name}.wav"
    if dest.exists():
        raise FileExistsError(f"a voice named '{name}' already exists")
    info = to_reference_wav(src, dest)
    return Voice(name, dest, info["duration"])


def rename_voice(old: str, new: str) -> Voice:
    new = safe_name(new)
    src = VOICES_DIR / f"{safe_name(old)}.wav"
    dest = VOICES_DIR / f"{new}.wav"
    if not src.exists():
        raise FileNotFoundError(f"no voice named '{old}'")
    if dest.exists():
        raise FileExistsError(f"a voice named '{new}' already exists")
    src.rename(dest)
    return Voice(new, dest, probe_duration(dest))


def delete_voice(name: str) -> None:
    p = VOICES_DIR / f"{safe_name(name)}.wav"
    if not p.exists():
        raise FileNotFoundError(f"no voice named '{name}'")
    p.unlink()


# ---------------------------------------------------------------- chunking

def chunk_text(text: str, max_chars: int = 400) -> list[str]:
    """Split on sentence boundaries so one long paragraph isn't one huge job."""
    text = (text or "").strip()
    if not text:
        return []
    # Protect against runaway single sentences.
    pieces = re.split(r"(?<=[.!?…])\s+|\n+", text)
    chunks, cur = [], ""
    for p in pieces:
        p = p.strip()
        if not p:
            continue
        if len(p) > max_chars:
            if cur:
                chunks.append(cur)
                cur = ""
            for i in range(0, len(p), max_chars):
                chunks.append(p[i:i + max_chars])
            continue
        if len(cur) + len(p) + 1 <= max_chars:
            cur = f"{cur} {p}".strip()
        else:
            chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks


def estimate_seconds(text: str, rtf: float = 6.5, cps: float = 15.0) -> float:
    """Rough wall-clock guess for the progress bar. ~15 chars/sec of speech."""
    audio_s = max(len(text or "") / cps, 0.5)
    return round(audio_s * rtf, 1)


# ---------------------------------------------------------------- jobs

@dataclass
class Job:
    id: str
    text: str
    voice: str
    language: str
    ref_text: str
    use_icl: bool
    status: str = "queued"
    progress: float = 0.0
    message: str = ""
    outputs: list = field(default_factory=list)
    created: float = field(default_factory=time.time)
    error: str = ""


_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def create_job(text, voice, language, ref_text="", use_icl=False) -> Job:
    job = Job(
        id=uuid.uuid4().hex[:12], text=text, voice=voice, language=language,
        ref_text=ref_text or "", use_icl=bool(use_icl),
    )
    with _lock:
        _jobs[job.id] = job
        # keep the table small
        if len(_jobs) > 60:
            for k in sorted(_jobs, key=lambda k: _jobs[k].created)[:20]:
                if _jobs[k].status in ("done", "error"):
                    _jobs.pop(k, None)
    return job


def get_job(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)


def update(job: Job, **kw) -> None:
    with _lock:
        for k, v in kw.items():
            setattr(job, k, v)


def run_job(job: Job) -> None:
    """Generate every chunk, concatenating into one file. Runs on a worker."""
    try:
        model = get_model()
        ref = VOICES_DIR / f"{safe_name(job.voice)}.wav"
        if not ref.exists():
            raise FileNotFoundError(f"voice '{job.voice}' no longer exists")

        update(job, status="preparing", progress=0.02,
               message="building voice prompt")
        if job.use_icl:
            prompt = model.create_voice_clone_prompt(
                ref_audio=str(ref), ref_text=job.ref_text or None,
                x_vector_only_mode=False,
            )
        else:
            prompt = model.create_voice_clone_prompt(
                ref_audio=str(ref), ref_text=None, x_vector_only_mode=True,
            )

        chunks = chunk_text(job.text)
        if not chunks:
            raise ValueError("nothing to speak - the text box was empty")
        total = len(chunks)
        sr = TARGET_SR
        import numpy as np
        collected, gaps = [], np.zeros(int(0.35 * sr), dtype=np.float32)

        for i, chunk in enumerate(chunks):
            done_chars = sum(len(c) for c in chunks[:i])
            frac = done_chars / max(len(job.text), 1)
            update(
                job, status="generating", progress=round(0.05 + 0.9 * frac, 3),
                message=f"part {i+1} of {total}",
            )
            wavs, sr = model.generate_voice_clone(
                text=chunk, voice_clone_prompt=prompt,
                language=job.language, non_streaming_mode=True,
            )
            for w in wavs:
                collected.append(w)
                collected.append(gaps)

        audio = np.concatenate(collected[:-1]) if collected else np.zeros(1, np.float32)

        stamp = time.strftime("%Y%m%d-%H%M%S")
        slug = re.sub(r"[^a-z0-9]+", "-", job.voice.lower()).strip("-") or "voice"
        out = OUTPUT_DIR / f"{stamp}-{slug}-{job.id[:6]}.wav"
        write_wav(out, audio, sr)

        update(job, status="done", progress=1.0, message="finished",
               outputs=[str(out)], error="")
    except Exception as e:  # noqa: BLE001 - surface anything to the UI
        update(job, status="error", progress=1.0, message="failed",
               error=f"{type(e).__name__}: {e}")


def list_outputs(limit: int = 40) -> list[Path]:
    files = sorted(OUTPUT_DIR.glob("*.wav"), key=lambda p: p.stat().st_mtime,
                   reverse=True)
    return files[:limit]


def start_worker(job: Job) -> Job:
    threading.Thread(target=run_job, args=(job,), daemon=True).start()
    return job
