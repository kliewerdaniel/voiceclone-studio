"""VoiceClone Studio - local Gradio UI."""
from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

import gradio as gr

import engine as E

AUDIO_EXTS = [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"]


# ---------------------------------------------------------------- helpers

def file_dialog_target() -> str:
    return str(E.VOICES_DIR)


# Linux: xdg-open handles Nautilus/Files/Thunar. macOS: open. Win: explorer.
_OPENER = {"darwin": ["open"], "win32": ["explorer"]}.get(
    sys.platform, ["xdg-open"])


def reveal(path: str) -> None:
    """Open the containing folder in the file manager."""
    p = Path(path)
    if not p.exists():
        return
    subprocess.Popen([*_OPENER, str(p.parent)], start_new_session=True)


def open_voices_folder() -> None:
    subprocess.Popen([*_OPENER, str(E.VOICES_DIR)], start_new_session=True)


def open_output_folder() -> None:
    subprocess.Popen([*_OPENER, str(E.OUTPUT_DIR)], start_new_session=True)


def flash(msg: str, error: bool = False) -> str:
    return f"⚠️ {msg}" if error else f"✅ {msg}"


# ---------------------------------------------------------------- voices tab

def _voice_table():
    rows = []
    for v in E.list_voices():
        rows.append([
            v.name,
            f"{v.duration:.1f} s" if v.duration else "—",
            f"{v.size_mb:.1f} MB",
        ])
    return rows


def _voice_choices():
    return [v.name for v in E.list_voices()]


def _preview(name):
    if not name:
        return None
    p = E.VOICES_DIR / f"{E.safe_name(name)}.wav"
    return str(p) if p.exists() else None


def upload_voice(files, name, progress=gr.Progress()):
    """files: list of uploaded paths. Normalises to mono 24kHz."""
    if not files:
        return _voice_table(), _voice_choices(), None, "⚠️ Choose an audio file first"
    src = Path(files[0])
    label = E.safe_name(name or src.stem)
    try:
        progress(0.2, desc="Converting to mono 24kHz")
        existing = {v.name for v in E.list_voices()}
        final = label
        n = 2
        while final in existing:
            final = f"{label} {n}"
            n += 1
        v = E.add_voice(final, src)
        progress(1.0, desc="Added")
        return (_voice_table(), _voice_choices(), str(v.path),
                flash(f"Voice '{v.name}' ready ({v.duration:.1f}s). "
                      f"Pick it on the Speak tab."))
    except Exception as e:  # noqa: BLE001
        return _voice_table(), _voice_choices(), None, flash(str(e), error=True)


def do_rename(old, new, confirm=gr.Progress()):
    if not old:
        return _voice_table(), _voice_choices(), "⚠️ Select a voice to rename"
    if not (new or "").strip():
        return _voice_table(), _voice_choices(), flash("New name is empty", True)
    try:
        v = E.rename_voice(old, new)
        return (_voice_table(), _voice_choices(),
                flash(f"Renamed to '{v.name}'"))
    except Exception as e:  # noqa: BLE001
        return _voice_table(), _voice_choices(), flash(str(e), error=True)


def do_delete(name, confirm=gr.Progress()):
    if not name:
        return _voice_table(), _voice_choices(), "⚠️ Select a voice to delete"
    try:
        E.delete_voice(name)
        return _voice_table(), _voice_choices(), flash(f"Deleted '{name}'")
    except Exception as e:  # noqa: BLE001
        return _voice_table(), _voice_choices(), flash(str(e), error=True)


def refresh_voices():
    return _voice_table(), _voice_choices(), _preview(None)


# ---------------------------------------------------------------- speak tab

def estimate(text):
    secs = E.estimate_seconds(text)
    if not (text or "").strip():
        return "Type something to get a time estimate."
    if secs > 600:
        return (f"⏱ Roughly {secs/60:.0f} min on this CPU. "
                f"That's long — consider splitting the text.")
    return f"⏱ Roughly {secs:.0f}s of generation (this box is ~6.5x slower than real time)."


def start_synthesis(text, voice, language, ref_text, use_icl):
    if not (text or "").strip():
        return None, "⚠️ Enter some text to speak", ""
    if not voice:
        return None, "⚠️ Add a voice on the Voices tab first", ""
    if len(text) > E.MAX_CHARS:
        return None, (f"⚠️ Text is {len(text)} chars, limit is {E.MAX_CHARS}. "
                      f"Trim it or split it into pieces."), ""
    if use_icl and not (ref_text or "").strip():
        return None, ("⚠️ High-quality mode needs the exact words spoken in the "
                      "reference recording, or turn off 'high quality'. "
                      "The default fast mode doesn't need it."), ""

    global ACTIVE_JOB
    job = E.create_job(text, voice, language, ref_text, use_icl)
    E.start_worker(job)
    ACTIVE_JOB = job.id
    return (job.id,
            f"⏱ Roughly {E.estimate_seconds(text):.0f}s. Working in the background…",
            "")


# The active job lives at module level, not in a gr.State, because
# gr.Timer ticks do not reliably share State scope with click handlers.
ACTIVE_JOB: str | None = None


def poll(job_id):
    """Return (progress_fraction, status_markdown, error_text, audio_path)."""
    if not job_id:
        return 0.0, "", "", None
    job = E.get_job(job_id)
    if not job:
        return 0.0, "⚠️ Job not found", "", None

    label = {
        "queued": "Queued",
        "preparing": "Loading model / building voice prompt",
        "generating": "Generating speech",
        "done": "Done",
        "error": "Failed",
    }.get(job.status, job.status)

    if job.status == "done" and job.outputs:
        return 1.0, f"✅ {label} — {job.message}", "", job.outputs[0]
    if job.status == "error":
        return 1.0, f"❌ {label}", job.error, None
    return job.progress, f"⏳ {label} — {job.message}", "", None


def err_box(msg: str):
    """Show the error field only when there is something to show."""
    if msg:
        return gr.update(value=msg, visible=True)
    return gr.update(value="", visible=False)


def progress_html(frac: float, active: bool) -> str:
    """Gradio 6 won't let a Timer write to gr.Progress, so draw the bar."""
    if not active or frac <= 0:
        return ""
    pct = max(0, min(100, int(frac * 100)))
    return (
        f'<div style="height:10px;background:#e5e7eb;border-radius:6px;'
        f'overflow:hidden;margin:6px 0 10px">'
        f'<div style="height:100%;width:{pct}%;background:#f97316;'
        f'border-radius:6px;transition:width .4s"></div></div>'
        f'<div style="font-size:0.85em;opacity:0.7">{pct}%</div>'
    )


def _fmt_out(p: Path) -> list:
    dur = E.probe_duration(p)
    return [p.name, time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime)),
            f"{dur:.1f}s" if dur else "—"]


def refresh_outputs():
    # Must be a list of ROWS (list of lists), not a list of row-lists.
    return [_fmt_out(p) for p in E.list_outputs()]


# ---------------------------------------------------------------- theme

CSS = """
.vc-h1 {font-size: 1.4em; font-weight: 600; margin-bottom: 0.1em;}
.vc-sub {opacity: 0.7; font-size: 0.92em; margin-bottom: 0.8em;}
footer {display: none !important;}
"""


def build():
    with gr.Blocks(title="VoiceClone Studio") as demo:
        demo.theme = gr.themes.Soft()
        demo.css = CSS
        gr.HTML('<div class="vc-h1">🎙️ VoiceClone Studio</div>'
                '<div class="vc-sub">Local, CPU-based voice cloning. '
                'Nothing leaves this machine.</div>')

        with gr.Tab("🎧 Voices"):
            gr.Markdown("Add a reference recording, then use it on the **Speak** tab. "
                        "Longer and cleaner audio (10–20s of one voice, no music) "
                        "gives the best clone. Files are converted to mono 24kHz "
                        f"automatically. Folder: `{E.VOICES_DIR}`")
            with gr.Row():
                with gr.Column(scale=3):
                    vtable = gr.Dataframe(
                        headers=["Name", "Length", "Size"],
                        datatype=["str", "str", "str"],
                        row_count=(6, "max"), interactive=False,
                        value=_voice_table(), label="Installed voices")
                with gr.Column(scale=2):
                    gr.Markdown("#### Preview")
                    vpreview = gr.Audio(label="", interactive=False)
                    vsel = gr.Dropdown(choices=_voice_choices(), label="Select a voice",
                                       value=(_voice_choices() or [None])[0])
                    vmsg = gr.Markdown("")
                    with gr.Row():
                        with gr.Column():
                            newname = gr.Textbox(label="Rename to", placeholder="new name")
                            rename_btn = gr.Button("Rename", variant="secondary")
                        with gr.Column():
                            gr.Markdown("<div style='height:26px'></div>")
                            delete_btn = gr.Button("Delete", variant="stop")
                    with gr.Row():
                        open_voices_btn = gr.Button("📁 Open voices folder", variant="primary")
                        refresh_v_btn = gr.Button("↻ Refresh", variant="secondary")

            gr.Markdown("#### Add a new voice")
            with gr.Row():
                vupload = gr.File(label="Reference audio (wav/mp3/m4a/flac/ogg)",
                                  file_types=AUDIO_EXTS, type="filepath")
                with gr.Column():
                    vname = gr.Textbox(label="Voice name", placeholder="chris")
                    add_btn = gr.Button("➕ Add voice", variant="primary")

        with gr.Tab("🗣️ Speak"):
            with gr.Row():
                with gr.Column(scale=3):
                    stext = gr.Textbox(
                        label="Text to speak", lines=8,
                        placeholder="Type what the voice should say…",
                        max_lines=20)
                    est = gr.Markdown(estimate(""))
                with gr.Column(scale=2):
                    svoice = gr.Dropdown(choices=_voice_choices(), label="Voice",
                                         value=(_voice_choices() or [None])[0])
                    slang = gr.Dropdown(choices=E.LANGUAGES, label="Language",
                                        value="English")
                    gr.Markdown("#### Quality")
                    use_icl = gr.Checkbox(
                        label="High quality (needs reference transcript)",
                        value=False,
                        info="Slower and closer to the original, but you must "
                             "type exactly what is said in the reference recording. "
                             "Off = fast mode, no transcript needed.")
                    ref_text = gr.Textbox(label="Reference transcript",
                                          lines=3, visible=False,
                                          placeholder="The exact words spoken in the voice file")
                    gen_btn = gr.Button("🔊 Generate", variant="primary", size="lg")
            prog = gr.HTML("")
            status = gr.Markdown("")
            err = gr.Textbox(label="Error", visible=False, interactive=False)
            player = gr.Audio(label="Result", interactive=False, visible=True)
            with gr.Row():
                reveal_btn = gr.Button("📂 Show file", variant="secondary")
                save_btn = gr.Button("💾 Save as…", variant="secondary")

        with gr.Tab("📼 Output") as out_tab:
            gr.Markdown(f"Generated clips in `{E.OUTPUT_DIR}`")
            otable = gr.Dataframe(headers=["File", "Created", "Length"],
                                  datatype=["str", "str", "str"],
                                  row_count=(8, "max"), interactive=False,
                                  value=refresh_outputs())
            oplayer = gr.Audio(label="Playback", interactive=False)
            osel = gr.Dropdown(choices=[], label="Choose a clip")
            with gr.Row():
                oopen_btn = gr.Button("📁 Open output folder", variant="primary")
                orefresh_btn = gr.Button("↻ Refresh", variant="secondary")

        # ---------------------------------------------------------- wiring
        vsel.change(_preview, inputs=vsel, outputs=vpreview)
        add_btn.click(upload_voice,
                      inputs=[vupload, vname],
                      outputs=[vtable, svoice, vpreview, vmsg])
        rename_btn.click(do_rename, inputs=[vsel, newname], outputs=[vtable, svoice, vmsg])
        delete_btn.click(do_delete, inputs=[vsel], outputs=[vtable, svoice, vmsg])
        refresh_v_btn.click(refresh_voices, outputs=[vtable, svoice, vpreview])
        open_voices_btn.click(None, js="() => { window.open('file://" + str(E.VOICES_DIR) + "'); return []; }")

        stext.input(estimate, inputs=stext, outputs=est)
        use_icl.change(lambda v: gr.update(visible=v), inputs=use_icl, outputs=ref_text)
        job_state = gr.State(None)
        gen_btn.click(start_synthesis,
                      inputs=[stext, svoice, slang, ref_text, use_icl],
                      outputs=[gr.State(), status, gr.State()])

        def _tick():
            frac, msg, errmsg, audio = poll(ACTIVE_JOB)
            active = bool(msg) and not msg.startswith(("✅", "❌"))
            return progress_html(frac, active), msg, err_box(errmsg), audio

        gr.Timer(2.0).tick(_tick, outputs=[prog, status, err, player])

        def _refresh_and_play():
            return refresh_outputs(), [p.name for p in E.list_outputs()]

        # Refresh the Output tab when the user opens it, rather than reacting
        # to the Speak tab's player (that fired spurious errors).
        out_tab.select(_refresh_and_play, inputs=None, outputs=[otable, osel])
        def _do_reveal(v):
            if v:
                reveal(v)
            return None

        reveal_btn.click(_do_reveal, inputs=player, outputs=None)

        def _do_save(v):
            if not v:
                return gr.update(visible=False)
            src = Path(v)
            dest = Path.home() / "Music" / src.name
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
            except Exception as e:  # noqa: BLE001
                savemsg = gr.update(value=f"⚠️ {e}", visible=True)
                return savemsg
            return gr.update(value=f"✅ Saved to {dest}", visible=True)

        savemsg = gr.Markdown("")
        save_btn.click(_do_save, inputs=player, outputs=savemsg)

        osel.change(lambda n: str(E.OUTPUT_DIR / n) if n else None,
                    inputs=osel, outputs=oplayer)
        orefresh_btn.click(lambda: (refresh_outputs(),
                                    [p.name for p in E.list_outputs()]),
                           outputs=[otable, osel])
        oopen_btn.click(None, js="() => { window.open('file://" + str(E.OUTPUT_DIR) + "'); return []; }")

    return demo


if __name__ == "__main__":
    app = build()
    print(f"[ui] model={E.CONFIG['model']}  http://127.0.0.1:7860", flush=True)
    app.queue(max_size=32).launch(
        server_name="127.0.0.1", server_port=7860,
        show_error=True, inbrowser=True, max_threads=4,
        favicon_path=None if not (E.APP_DIR / "favicon.ico").exists() else str(E.APP_DIR / "favicon.ico"),
    )
