# VoiceClone Studio

A local, offline voice-cloning studio. Give it one reference recording, type
some text, and it speaks in that voice.

Everything runs on your own machine — CPU only, no GPU required, no audio ever
leaves the box. The Gradio UI listens on `127.0.0.1` and nothing else.

![stack](https://img.shields.io/badge/python-3.10%2B-blue) ![license](https://img.shields.io/badge/license-MIT-green)

---

## What it does

- **🎧 Voices** — drop in a `.wav`/`.mp3`/`.m4a`/`.flac`/`.ogg`; it's
  auto-normalised to mono 24 kHz. Rename, preview, delete.
- **🗣️ Speak** — pick a voice, a language, type text. Generation runs in the
  background with a real progress bar, so the UI stays responsive.
- **📼 Output** — every clip is kept with a timestamped filename; play back,
  reveal in the file manager, or copy to `~/Music`.

Two quality modes:

| Mode | Needs the reference transcript? | Speed | Fidelity |
|---|---|---|---|
| **Fast** (default) | no | ~6.5x real time on CPU | good |
| **High quality** | yes — the exact words spoken | slower | closer to the original |

## How it works

- Model: [`Qwen/Qwen3-TTS-12Hz-0.6B-Base`](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base)
  (0.6B — about 1.2 GB, runs comfortably in ~4 GB of RAM). A 1.7B variant is
  available via `config.json` if you have ~9 GB to spare.
- The model is downloaded from Hugging Face on first run and cached locally;
  after that you can run fully offline.
- Long text is split on sentence boundaries (~400 chars), generated chunk by
  chunk with a 350 ms gap between chunks, then concatenated into one `.wav`.

## Requirements

- Python 3.10+ (developed on 3.12)
- `ffmpeg` / `ffprobe` on `$PATH`
  - Arch: `sudo pacman -S ffmpeg`
  - Debian/Ubuntu: `sudo apt install ffmpeg`
- ~2 GB free disk for the venv + model, ~4 GB RAM for the 0.6B model
- A first run downloads the model from Hugging Face (~1.2 GB)

## Install

```bash
git clone https://github.com/kliewerdaniel/voiceclone-studio.git
cd voiceclone-studio
python3 -m venv .venv

# CPU-only torch keeps the download to a few hundred MB.
# Skip this line if you want CUDA.
.venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements.txt

cp config.example.json config.json   # optional
```

## Run

```bash
./run.sh
```

It opens `http://127.0.0.1:7860` in your browser. `run.sh` is safe to run
repeatedly — if an instance is already up it just tells you the URL, and if a
stale one is holding the port it cleans it up first.

To use a different port:

```bash
VOICE_CLONE_PORT=8080 ./run.sh
```

To run without the wrapper:

```bash
.venv/bin/python app.py
```

### Gradio's own auth

The app is bound to loopback, so on a single-user desktop no login is needed.
If you expose it beyond that (reverse proxy, SSH tunnel, container port), put
your own auth in front of it — anyone who can reach the port can clone any
voice you have loaded.

## Getting a good clone

1. **10–20 seconds** of clean speech, one speaker, no music or background noise.
2. WAV or a lossless source if you have one; MP3 is fine.
3. Use **High quality** mode and type the reference transcript verbatim if you
   want the closest match.
4. More reference audio helps, but past ~30 s you gain very little.

## Layout

```
app.py                 Gradio UI: voices tab, speak tab, output tab
engine.py              model loading, voice library, chunked generation
run.sh                 launcher (port reuse + stale instance cleanup)
config.example.json    model choice + thread count
voices/                your reference recordings  (git-ignored)
output/                generated clips            (git-ignored)
models/                local weight dir           (git-ignored)
```

## Configuration

`config.json` (copy from `config.example.json`):

| Key | Default | Notes |
|---|---|---|
| `model` | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | or `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |
| `threads` | CPU count | capped at the physical core count |

## Troubleshooting

**Port 7860 already in use** — another instance is running; `run.sh` will
report it. Otherwise set `VOICE_CLONE_PORT`.

**`ffmpeg failed:` in the error box** — `ffmpeg` isn't on `$PATH`.

**First generation hangs for a minute** — that's the model loading. It only
happens once per session; the status line says so.

**Killed mid-generation** — out of RAM. Use the 0.6B model and close other
heavy apps, or move to the 1.7B model with more RAM available.

**Generation is very slow** — expected on CPU. The estimate under the text box
tells you roughly how long before you start. Roughly 6.5x slower than real
time on a 2017-era dual-core laptop; a modern 8-thread desktop does much
better.

**Torch errors about `nvidia` / `libcuda`** — you installed the CUDA build of
torch by accident. Reinstall from the CPU index (see Install).

## Ethical use

This is a tool for legitimate work: voicing your own projects, game and
documentary narration, accessibility, dubbing, prototyping. Cloning someone's
voice without their consent to deceive or defraud is illegal in many
jurisdictions, and it's just a rotten thing to do. Get permission.

Model weights are licensed by Alibaba (Apache-2.0); see the model card. This
repo's own code is MIT.

## Credits

- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) by Alibaba
- [Gradio](https://www.gradio.app/) for the UI
