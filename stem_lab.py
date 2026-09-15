from __future__ import annotations

import base64
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from structure_lab import (
    analyze_harmony,
    build_visual_blocks,
    combine_structure_candidates,
    detect_harmonic_motifs,
    detect_lyric_repeats,
    lyrics_phrases,
)
import mimetypes
import shutil
import subprocess
import sys
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
CACHE_DIR = APP_DIR / "stem_lab_data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEMUCS_MODEL = "htdemucs"
STEM_NAMES = ("vocals", "drums", "bass", "other")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_suffix(filename: str) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix in {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".aac", ".wma"}:
        return suffix
    return ".bin"


def audio_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime:
        return mime
    return "audio/wav" if path.suffix.lower() == ".wav" else "audio/mpeg"


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _preview_target(path: Path, preview_dir: Path) -> Path:
    preview_dir.mkdir(parents=True, exist_ok=True)
    return preview_dir / f"{path.stem}.mp3"


def _preview_is_current(source: Path, target: Path) -> bool:
    return (
        target.is_file()
        and target.stat().st_size > 0
        and target.stat().st_mtime_ns >= source.stat().st_mtime_ns
    )


def make_browser_preview(path: Path, preview_dir: Path) -> Path:
    """Build one compact MP3 copy used only by the browser mixer."""
    target = _preview_target(path, preview_dir)

    if _preview_is_current(path, target):
        return target

    if not ffmpeg_available():
        raise RuntimeError(
            "FFmpeg est requis pour créer les copies MP3 légères du lecteur."
        )

    # If the uploaded original is already MP3, copying it is instantaneous and
    # avoids an unnecessary fifth audio transcode.
    if path.suffix.lower() == ".mp3":
        shutil.copy2(path, target)
        return target

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vn",
        "-map_metadata",
        "-1",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "96k",
        "-ar",
        "44100",
        str(target),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not target.is_file():
        raise RuntimeError(
            "Impossible de créer la pré-écoute MP3 pour "
            f"{path.name}:\n{proc.stdout or ''}"
        )
    return target


def prepare_browser_previews(
    source: Path,
    stems: dict[str, Path],
) -> dict[str, Path]:
    """Prepare all browser files in parallel.

    The first R2 run looked blank while Streamlit synchronously transcoded four
    large WAV stems one after another. R3 does the work concurrently and shows
    an explicit status to the user.
    """
    preview_dir = source.parent / "browser_preview"
    inputs = {"original": source, **stems}

    result: dict[str, Path] = {}
    pending = {}

    # Reuse all current previews immediately.
    for name, path in inputs.items():
        target = _preview_target(path, preview_dir)
        if _preview_is_current(path, target):
            result[name] = target
        else:
            pending[name] = path

    if not pending:
        return result

    status = st.status(
        f"Préparation du lecteur : {len(pending)} piste(s) à compresser…",
        expanded=True,
    )
    progress = st.progress(0.0)
    done = 0

    workers = min(4, max(1, len(pending)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(make_browser_preview, path, preview_dir): name
            for name, path in pending.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            result[name] = future.result()
            done += 1
            progress.progress(done / len(pending))
            status.write(f"✓ {name}")

    progress.empty()
    status.update(
        label="Préparation du lecteur terminée.",
        state="complete",
        expanded=False,
    )
    return result



def write_uploaded_source(uploaded) -> tuple[str, Path]:
    data = uploaded.getvalue()
    audio_hash = sha256_bytes(data)
    work_dir = CACHE_DIR / audio_hash
    work_dir.mkdir(parents=True, exist_ok=True)
    source = work_dir / f"source{safe_suffix(uploaded.name)}"
    if not source.exists() or source.stat().st_size != len(data):
        source.write_bytes(data)
    return audio_hash, source


def demucs_available() -> bool:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "demucs", "--help"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
        )
        return proc.returncode == 0
    except Exception:
        return False


def demucs_output_dir(work_dir: Path, source: Path) -> Path:
    return work_dir / "demucs" / DEMUCS_MODEL / source.stem


def locate_stems(work_dir: Path, source: Path) -> dict[str, Path]:
    root = demucs_output_dir(work_dir, source)
    result = {}
    for name in STEM_NAMES:
        path = root / f"{name}.wav"
        if path.is_file():
            result[name] = path
    return result


def extract_stems(source: Path, work_dir: Path) -> tuple[dict[str, Path], str]:
    out_root = work_dir / "demucs"
    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "-n",
        DEMUCS_MODEL,
        "-o",
        str(out_root),
        str(source),
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    log = proc.stdout or ""
    stems = locate_stems(work_dir, source)

    if proc.returncode != 0:
        raise RuntimeError(
            f"Demucs a échoué (code {proc.returncode}).\n\n"
            + "\n".join(log.splitlines()[-30:])
        )

    missing = [name for name in STEM_NAMES if name not in stems]
    if missing:
        raise RuntimeError(
            "Extraction terminée mais stems manquants : "
            + ", ".join(missing)
            + "\n\n"
            + "\n".join(log.splitlines()[-30:])
        )

    return stems, log



# ---------------------------------------------------------------------------
# Speech / lyrics analysis
# ---------------------------------------------------------------------------

SPEECH_CACHE_DIRNAME = "speech_analysis"


def _speech_cache_path(work_dir: Path, model_name: str) -> Path:
    safe_model = "".join(
        ch if ch.isalnum() or ch in "._-" else "_"
        for ch in str(model_name)
    )
    root = work_dir / SPEECH_CACHE_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root / f"whisper_{safe_model}.json"


def load_speech_cache(work_dir: Path, model_name: str) -> dict[str, Any] | None:
    path = _speech_cache_path(work_dir, model_name)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_speech_cache(
    work_dir: Path,
    model_name: str,
    payload: dict[str, Any],
) -> Path:
    path = _speech_cache_path(work_dir, model_name)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


@st.cache_resource(show_spinner=False)
def _load_whisper_model(model_name: str):
    import whisper
    return whisper.load_model(model_name)


def whisper_available() -> bool:
    try:
        import whisper  # noqa: F401
        return True
    except Exception:
        return False


def torch_device_label() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return f"CUDA · {torch.cuda.get_device_name(0)}"
    except Exception:
        pass
    return "CPU"


def _extract_whisper_words(result: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment in result.get("segments", []) or []:
        for item in segment.get("words", []) or []:
            text = str(item.get("word", "") or "").strip()
            if not text:
                continue
            words.append({
                "start": round(float(item.get("start", 0.0) or 0.0), 3),
                "end": round(float(item.get("end", 0.0) or 0.0), 3),
                "text": text,
            })
    return words


def transcribe_vocals(
    vocals_path: Path,
    *,
    model_name: str,
) -> dict[str, Any]:
    """Transcribe the isolated vocal stem with Whisper word timestamps."""
    import torch

    model = _load_whisper_model(model_name)
    use_fp16 = bool(torch.cuda.is_available())

    result = model.transcribe(
        str(vocals_path),
        task="transcribe",
        word_timestamps=True,
        fp16=use_fp16,
        verbose=False,
    )

    payload = {
        "engine": "openai-whisper",
        "model": str(model_name),
        "source": "demucs-vocals",
        "language": str(result.get("language", "") or ""),
        "text": str(result.get("text", "") or "").strip(),
        "words": _extract_whisper_words(result),
    }
    return payload


def phonemizer_available() -> bool:
    try:
        from phonemizer.backend import EspeakBackend
        return bool(EspeakBackend.is_available())
    except Exception:
        return False


def _language_to_espeak(language: str) -> str:
    lang = str(language or "").lower()
    mapping = {
        "fr": "fr-fr",
        "en": "en-us",
        "it": "it",
        "es": "es",
        "de": "de",
        "pt": "pt",
        "nl": "nl",
    }
    return mapping.get(lang, lang or "fr-fr")


def add_phonemes(
    words: list[dict[str, Any]],
    *,
    language: str,
) -> tuple[list[dict[str, Any]], str | None]:
    """Attach IPA phonemes to each Whisper word using eSpeak/phonemizer."""
    if not words:
        return [], None

    try:
        from phonemizer.backend import EspeakBackend
        from phonemizer.separator import Separator

        if not EspeakBackend.is_available():
            return words, "eSpeak/phonemizer indisponible"

        backend = EspeakBackend(
            language=_language_to_espeak(language),
            preserve_punctuation=True,
            with_stress=True,
        )
        texts = [str(item["text"]) for item in words]
        phones = backend.phonemize(
            texts,
            separator=Separator(phone=" ", word=None),
            strip=True,
            njobs=1,
        )

        enriched = []
        for item, phone in zip(words, phones):
            row = dict(item)
            row["phonemes"] = str(phone or "").strip()
            enriched.append(row)
        return enriched, None
    except Exception as exc:
        return words, f"Phonèmes indisponibles : {exc}"


def translator_available() -> bool:
    try:
        import deep_translator  # noqa: F401
        return True
    except Exception:
        return False


def translate_text(
    text: str,
    *,
    source_language: str,
    target_language: str,
) -> tuple[str, str | None]:
    """Translate lyrics text with deep-translator/GoogleTranslator.

    This step uses the internet. It is intentionally independent from Whisper
    so the transcription source remains the isolated vocal stem.
    """
    text = str(text or "").strip()
    if not text:
        return "", None

    if not translator_available():
        return "", "deep-translator n'est pas installé"

    try:
        from deep_translator import GoogleTranslator

        source = str(source_language or "auto").lower()
        target = str(target_language or "fr").lower()
        translated = GoogleTranslator(
            source=source if source else "auto",
            target=target,
        ).translate(text)
        return str(translated or "").strip(), None
    except Exception as exc:
        return "", f"Traduction indisponible : {exc}"


def _words_dataframe_rows(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in words:
        rows.append({
            "Début": float(item.get("start", 0.0) or 0.0),
            "Fin": float(item.get("end", 0.0) or 0.0),
            "Mot": str(item.get("text", "") or ""),
            "Phonèmes IPA": str(item.get("phonemes", "") or ""),
        })
    return rows



def latest_cached_speech_payload(work_dir: Path) -> dict[str, Any] | None:
    """Return the best cached Whisper payload for downstream structure analysis.

    Prefer the most complete timeline (latest word end + word count), then mtime.
    This avoids accidentally selecting an older/partial cache just because its
    model name happens to sort first.
    """
    root = work_dir / SPEECH_CACHE_DIRNAME
    if not root.is_dir():
        return None

    candidates = []
    for path in root.glob("whisper_*.json"):
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            words = list(payload.get("words", []) or [])
            if not words:
                continue
            valid_words = []
            latest_end = 0.0
            for item in words:
                text = str(item.get("text", "") or "").strip()
                if not text:
                    continue
                start = float(item.get("start", 0.0) or 0.0)
                end = float(item.get("end", start) or start)
                # Keep zero-duration items out of downstream structure analysis.
                if end <= start:
                    continue
                valid_words.append(item)
                latest_end = max(latest_end, end)

            if not valid_words:
                continue

            candidates.append({
                "path": path,
                "payload": payload,
                "valid_words": valid_words,
                "latest_end": latest_end,
                "word_count": len(valid_words),
                "mtime": path.stat().st_mtime_ns,
            })
        except Exception:
            continue

    if not candidates:
        return None

    # Coverage first, then number of usable words, then recency.
    candidates.sort(
        key=lambda item: (
            float(item["latest_end"]),
            int(item["word_count"]),
            int(item["mtime"]),
        ),
        reverse=True,
    )
    best = candidates[0]
    payload = dict(best["payload"])
    payload["words"] = best["valid_words"]
    payload["_cache_path"] = str(best["path"])
    payload["_coverage_end"] = float(best["latest_end"])
    payload["_usable_word_count"] = int(best["word_count"])
    return payload


def latest_cached_words(work_dir: Path) -> list[dict[str, Any]]:
    payload = latest_cached_speech_payload(work_dir)
    if not payload:
        return []
    return list(payload.get("words", []) or [])


def render_speech_analysis(
    *,
    work_dir: Path,
    stems: dict[str, Path],
) -> None:
    st.subheader("Analyse paroles / phonèmes / traduction")
    st.caption(
        "Source de transcription : stem `vocals.wav` Demucs. "
        "Les timestamps restent relatifs à l'audio original."
    )

    vocals = stems.get("vocals")
    if vocals is None or not vocals.is_file():
        st.warning("Stem vocal absent.")
        return

    if not whisper_available():
        st.error(
            "Whisper n'est pas installé. "
            "Installer avec : `python -m pip install openai-whisper`"
        )
        return

    col_model, col_target = st.columns(2)
    with col_model:
        model_name = st.selectbox(
            "Modèle Whisper",
            ["tiny", "base", "small"],
            index=1,
            help=(
                "base = bon compromis. small = plus précis mais plus lent. "
                "Le modèle est mis en cache par Streamlit."
            ),
        )
    with col_target:
        target_label = st.selectbox(
            "Traduction vers",
            ["Français", "Anglais", "Italien", "Espagnol", "Allemand"],
            index=0,
        )

    target_codes = {
        "Français": "fr",
        "Anglais": "en",
        "Italien": "it",
        "Espagnol": "es",
        "Allemand": "de",
    }
    target_code = target_codes[target_label]

    st.caption(
        f"Whisper : {torch_device_label()} · "
        f"phonèmes eSpeak : {'OK' if phonemizer_available() else 'indisponible'} · "
        f"traduction : {'OK' if translator_available() else 'indisponible'}"
    )

    cached = load_speech_cache(work_dir, model_name)

    analyse_clicked = st.button(
        "Analyser les paroles",
        type="primary",
        width="stretch",
        key=f"speech_analyse_{model_name}",
    )

    if analyse_clicked:
        with st.spinner("Whisper analyse le stem vocal…"):
            payload = transcribe_vocals(
                vocals,
                model_name=model_name,
            )
            words, phoneme_error = add_phonemes(
                list(payload.get("words", []) or []),
                language=str(payload.get("language", "") or ""),
            )
            payload["words"] = words
            payload["phoneme_error"] = phoneme_error
            save_speech_cache(work_dir, model_name, payload)
            cached = payload

    if not cached:
        st.info("Clique `Analyser les paroles`.")
        return

    source_language = str(cached.get("language", "") or "auto")
    lyrics = str(cached.get("text", "") or "").strip()
    words = list(cached.get("words", []) or [])

    st.success(
        f"Transcription disponible · langue détectée : "
        f"`{source_language}` · {len(words)} mots."
    )

    tab_text, tab_words, tab_translation = st.tabs(
        ["Paroles", "Mots + phonèmes", "Traduction"]
    )

    with tab_text:
        st.text_area(
            "Texte transcrit",
            value=lyrics,
            height=260,
            key=f"lyrics_text_{model_name}",
        )

    with tab_words:
        if cached.get("phoneme_error"):
            st.warning(str(cached.get("phoneme_error")))
        if words:
            st.dataframe(
                _words_dataframe_rows(words),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("Aucun mot horodaté.")

    with tab_translation:
        if not translator_available():
            st.warning(
                "Installer `deep-translator` pour activer cette fonction."
            )
        elif source_language == target_code:
            st.info("La langue source et la langue cible sont identiques.")
            st.text_area(
                "Traduction",
                value=lyrics,
                height=260,
                key=f"translation_same_{model_name}_{target_code}",
            )
        else:
            translation_key = (
                f"_stem_lab_translation_{model_name}_{source_language}_{target_code}"
            )
            if st.button(
                f"Traduire vers {target_label}",
                key=f"translate_{model_name}_{target_code}",
            ):
                with st.spinner("Traduction…"):
                    translated, error = translate_text(
                        lyrics,
                        source_language=source_language,
                        target_language=target_code,
                    )
                    st.session_state[translation_key] = {
                        "text": translated,
                        "error": error,
                    }

            translation_state = st.session_state.get(translation_key, {})
            if translation_state.get("error"):
                st.warning(str(translation_state["error"]))
            if translation_state.get("text"):
                st.text_area(
                    "Traduction",
                    value=str(translation_state["text"]),
                    height=260,
                    key=f"translation_text_{model_name}_{target_code}",
                )

    export_payload = {
        "language": source_language,
        "text": lyrics,
        "words": words,
    }
    st.download_button(
        "⬇ Exporter l'analyse paroles JSON",
        data=json.dumps(
            export_payload,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8"),
        file_name="speech_analysis.json",
        mime="application/json",
        width="stretch",
    )


_PLAYER_HTML = """
<div class="stem-player">
  <div class="transport">
    <button class="play" type="button">▶ Lecture</button>
    <button class="pause" type="button">⏸ Pause</button>
    <button class="stop" type="button">⏹ Stop</button>
    <span class="time">0:00 / 0:00</span>
  </div>

  <input class="seek" type="range" min="0" max="1" step="0.001" value="0">

  <div class="tracks"></div>

  <div class="lyrics-wrap">
    <div class="lyrics-title">Paroles synchronisées</div>
    <div class="lyrics-strip">
      <div class="lyrics-track"></div>
    </div>
  </div>

  <div class="hint">
    WebAudio : une horloge unique pour toutes les pistes. Aucun micro-seek pendant la lecture.
  </div>
</div>
"""

_PLAYER_CSS = """
:host {
  display:block;
  width:100%;
}
.stem-player {
  box-sizing:border-box;
  width:100%;
  border:1px solid color-mix(in srgb, var(--st-text-color) 25%, transparent);
  border-radius:10px;
  padding:12px;
  background:color-mix(in srgb, var(--st-text-color) 4%, transparent);
  color:var(--st-text-color);
  font-family:var(--st-font);
}
.transport {
  display:flex;
  align-items:center;
  gap:8px;
  flex-wrap:wrap;
}
.transport button {
  min-height:34px;
  border-radius:7px;
  border:1px solid color-mix(in srgb, var(--st-text-color) 35%, transparent);
  background:color-mix(in srgb, var(--st-text-color) 8%, transparent);
  color:var(--st-text-color);
  cursor:pointer;
  padding:5px 10px;
}
.time {
  margin-left:auto;
  font-variant-numeric:tabular-nums;
  font-size:12px;
  opacity:.75;
}
.seek {
  width:100%;
  margin:10px 0 12px;
}
.tracks {
  display:grid;
  gap:8px;
}
.track {
  display:grid;
  grid-template-columns:minmax(90px, 130px) 74px 1fr;
  gap:10px;
  align-items:center;
}
.track-name {
  font-weight:700;
}
.track-toggle {
  display:flex;
  align-items:center;
  gap:5px;
  font-size:12px;
}
.track input[type="range"] {
  width:100%;
}
.lyrics-wrap {
  margin-top:14px;
  padding-top:10px;
  border-top:1px solid color-mix(in srgb, var(--st-text-color) 18%, transparent);
}
.lyrics-title {
  font-size:12px;
  font-weight:700;
  opacity:.8;
  margin-bottom:6px;
}
.lyrics-strip {
  position:relative;
  overflow:hidden;
  min-height:62px;
  border-radius:8px;
  background:color-mix(in srgb, var(--st-text-color) 5%, transparent);
}
.lyrics-track {
  position:absolute;
  left:50%;
  top:50%;
  transform:translate(0,-50%);
  white-space:nowrap;
  transition:transform 80ms linear;
}
.lyric-word {
  display:inline-block;
  margin:0 5px;
  font-size:18px;
  opacity:.30;
  transition:opacity 80ms linear, transform 80ms linear;
}
.lyric-word.past {
  opacity:.48;
}
.lyric-word.current {
  opacity:1;
  font-weight:800;
  transform:scale(1.08);
}
.hint {
  margin-top:10px;
  font-size:11px;
  opacity:.68;
}
@media(max-width:650px) {
  .track {
    grid-template-columns:1fr;
  }
  .time {
    width:100%;
    margin-left:0;
  }
}
"""

_PLAYER_JS = r"""
export default function(component) {
  const data = component.data || {};
  const root = component.parentElement;

  const playButton = root.querySelector(".play");
  const pauseButton = root.querySelector(".pause");
  const stopButton = root.querySelector(".stop");
  const seek = root.querySelector(".seek");
  const timeLabel = root.querySelector(".time");
  const tracksNode = root.querySelector(".tracks");
  const lyricsWrap = root.querySelector(".lyrics-wrap");
  const lyricsStrip = root.querySelector(".lyrics-strip");
  const lyricsTrack = root.querySelector(".lyrics-track");

  const defs = Array.isArray(data.tracks) ? data.tracks : [];
  const words = Array.isArray(data.words) ? data.words : [];

  let context = null;
  let decoded = [];
  let gains = [];
  let sources = [];
  let ready = false;
  let playing = false;
  let disposed = false;
  let position = 0;
  let startedAtContextTime = 0;
  let duration = 0;
  let raf = null;
  let activeWordIndex = -1;

  function fmt(seconds) {
    seconds = Math.max(0, Number(seconds) || 0);
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return m + ":" + String(s).padStart(2, "0");
  }

  function currentTime() {
    if (!playing || !context) return position;
    return Math.max(0, Math.min(duration, position + (context.currentTime - startedAtContextTime)));
  }

  function decodeBase64(base64) {
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes.buffer;
  }

  async function ensureReady() {
    if (ready) {
      if (context.state === "suspended") await context.resume();
      return;
    }

    playButton.disabled = true;
    playButton.textContent = "Chargement audio…";

    context = new (window.AudioContext || window.webkitAudioContext)({
      latencyHint: "interactive"
    });

    decoded = [];
    gains = [];

    for (let i = 0; i < defs.length; i += 1) {
      const buffer = await context.decodeAudioData(
        decodeBase64(String(defs[i].base64 || ""))
      );
      decoded.push(buffer);

      const gain = context.createGain();
      gain.gain.value = defs[i].enabled ? Number(defs[i].volume ?? 0.8) : 0;
      gain.connect(context.destination);
      gains.push(gain);

      if (i === 0) {
        duration = Number(buffer.duration || 0);
      }
    }

    seek.max = String(Math.max(0.001, duration));
    ready = true;
    playButton.disabled = false;
    playButton.textContent = "▶ Lecture";
  }

  function stopSources() {
    sources.forEach((source) => {
      try { source.stop(); } catch (_) {}
      try { source.disconnect(); } catch (_) {}
    });
    sources = [];
  }

  function startSources(offset) {
    stopSources();

    const when = context.currentTime + 0.030;
    sources = decoded.map((buffer, index) => {
      const source = context.createBufferSource();
      source.buffer = buffer;
      source.connect(gains[index]);

      const safeOffset = Math.max(0, Math.min(Number(offset) || 0, Math.max(0, buffer.duration - 0.001)));
      source.start(when, safeOffset);
      return source;
    });

    position = Math.max(0, Math.min(duration, Number(offset) || 0));
    startedAtContextTime = when;
    playing = true;
  }

  async function playAll() {
    await ensureReady();
    if (context.state === "suspended") await context.resume();

    if (playing) return;

    if (position >= duration - 0.01) {
      position = 0;
    }

    startSources(position);
  }

  function pauseAll() {
    if (!playing) return;
    position = currentTime();
    playing = false;
    stopSources();
  }

  function stopAll() {
    playing = false;
    stopSources();
    position = 0;
    seek.value = "0";
    renderLyrics(0);
    timeLabel.textContent = "0:00 / " + fmt(duration);
  }

  function seekTo(time) {
    const t = Math.max(0, Math.min(duration, Number(time) || 0));
    position = t;

    if (playing) {
      startSources(t);
    }

    renderLyrics(t);
  }

  // Build volume controls against GainNodes.
  defs.forEach((track, index) => {
    const row = document.createElement("div");
    row.className = "track";

    const name = document.createElement("div");
    name.className = "track-name";
    name.textContent = String(track.label || track.name || "Track");

    const toggleWrap = document.createElement("label");
    toggleWrap.className = "track-toggle";

    const toggle = document.createElement("input");
    toggle.type = "checkbox";
    toggle.checked = Boolean(track.enabled);

    const toggleText = document.createElement("span");
    toggleText.textContent = "Actif";

    toggleWrap.append(toggle, toggleText);

    const volume = document.createElement("input");
    volume.type = "range";
    volume.min = "0";
    volume.max = "1";
    volume.step = "0.01";
    volume.value = String(
      track.volume === undefined ? 0.8 : Number(track.volume)
    );

    toggle.addEventListener("change", () => {
      if (ready && gains[index]) {
        gains[index].gain.value = toggle.checked ? Number(volume.value) : 0;
      }
    });

    volume.addEventListener("input", () => {
      if (ready && gains[index] && toggle.checked) {
        gains[index].gain.value = Number(volume.value);
      }
    });

    row.append(name, toggleWrap, volume);
    tracksNode.appendChild(row);
  });

  // Lyrics nodes.
  const lyricNodes = words.map((word) => {
    const span = document.createElement("span");
    span.className = "lyric-word";
    span.textContent = String(word.text || "").trim();
    lyricsTrack.appendChild(span);
    return span;
  });

  if (!words.length) {
    lyricsWrap.style.display = "none";
  }

  function findWordIndex(time) {
    if (!words.length) return -1;
    let low = 0;
    let high = words.length - 1;
    let answer = 0;
    while (low <= high) {
      const middle = (low + high) >> 1;
      if (Number(words[middle].start || 0) <= time) {
        answer = middle;
        low = middle + 1;
      } else {
        high = middle - 1;
      }
    }
    return answer;
  }

  function renderLyrics(time) {
    if (!words.length) return;
    const index = findWordIndex(time);
    if (index < 0 || !lyricNodes[index]) return;

    if (index !== activeWordIndex) {
      activeWordIndex = index;
      lyricNodes.forEach((node, i) => {
        node.classList.toggle("past", i < index);
        node.classList.toggle("current", i === index);
      });
    }

    const current = lyricNodes[index];
    const next = lyricNodes[index + 1];
    const currentCenter = current.offsetLeft + current.offsetWidth / 2;
    let targetCenter = currentCenter;

    if (next) {
      const start = Number(words[index].start || 0);
      const nextStart = Math.max(
        start + 0.04,
        Number(words[index + 1].start || start + 0.5)
      );
      const progress = Math.max(
        0,
        Math.min(1, (time - start) / (nextStart - start))
      );
      const nextCenter = next.offsetLeft + next.offsetWidth / 2;
      targetCenter = currentCenter + (nextCenter - currentCenter) * progress;
    }

    lyricsTrack.style.transform =
      "translate(" + (lyricsStrip.clientWidth / 2 - targetCenter) + "px,-50%)";
  }

  function tick() {
    if (disposed) return;

    const t = currentTime();
    seek.value = String(t);
    timeLabel.textContent = fmt(t) + " / " + fmt(duration);
    renderLyrics(t);

    if (playing && t >= duration - 0.01) {
      stopAll();
    }

    raf = requestAnimationFrame(tick);
  }

  playButton.addEventListener("click", playAll);
  pauseButton.addEventListener("click", pauseAll);
  stopButton.addEventListener("click", stopAll);

  seek.addEventListener("input", () => {
    seekTo(Number(seek.value || 0));
  });

  tick();

  return function() {
    disposed = true;
    if (raf !== null) cancelAnimationFrame(raf);
    stopSources();
    try {
      if (context) context.close();
    } catch (_) {}
  };
}
"""

_STEM_PLAYER = st.components.v2.component(
    "ezscore_stem_lab_player",
    html=_PLAYER_HTML,
    css=_PLAYER_CSS,
    js=_PLAYER_JS,
    isolate_styles=True,
)


def render_player(
    source: Path,
    stems: dict[str, Path],
    key: str,
    words: list[dict[str, Any]] | None = None,
) -> None:
    st.caption(
        "Préparation/chargement du lecteur… "
        "La première ouverture peut prendre quelques secondes."
    )

    previews = prepare_browser_previews(source, stems)
    tracks = []

    def pack(
        name: str,
        label: str,
        path: Path,
        enabled: bool,
        volume: float,
    ):
        preview = previews[name]
        tracks.append({
            "name": name,
            "label": label,
            "mime": "audio/mpeg",
            "base64": base64.b64encode(preview.read_bytes()).decode("ascii"),
            "enabled": enabled,
            "volume": volume,
            "source_bytes": int(path.stat().st_size),
            "preview_bytes": int(preview.stat().st_size),
        })

    pack("original", "Original", source, True, 0.75)

    defaults = {
        "vocals": (True, 0.90),
        "drums": (False, 0.75),
        "bass": (False, 0.75),
        "other": (False, 0.75),
    }
    labels = {
        "vocals": "Chant",
        "drums": "Batterie",
        "bass": "Basse",
        "other": "Other",
    }

    for name in STEM_NAMES:
        if name in stems:
            enabled, volume = defaults[name]
            pack(name, labels[name], stems[name], enabled, volume)

    payload_bytes = sum(len(track["base64"]) for track in tracks)
    st.caption(
        "Lecteur WebAudio prêt · previews MP3 96 kb/s · "
        f"payload ≈ {payload_bytes / 1024 / 1024:.1f} Mio · "
        "WAV Demucs intacts."
    )

    _STEM_PLAYER(
        data={
            "tracks": tracks,
            "words": list(words or []),
        },
        key=key,
        width="stretch",
        height=440,
    )



def download_stem_buttons(stems: dict[str, Path]) -> None:
    cols = st.columns(len(STEM_NAMES))
    labels = {
        "vocals": "Chant",
        "drums": "Batterie",
        "bass": "Basse",
        "other": "Other",
    }

    for col, name in zip(cols, STEM_NAMES):
        with col:
            path = stems.get(name)
            if path is None:
                st.button(
                    f"{labels[name]} absent",
                    disabled=True,
                    width="stretch",
                    key=f"missing_{name}",
                )
            else:
                st.download_button(
                    f"⬇ {labels[name]}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="audio/wav",
                    width="stretch",
                    key=f"download_{name}_{path.stat().st_size}",
                )



def render_structure_analysis(
    *,
    work_dir: Path,
    stems: dict[str, Path],
) -> None:
    st.subheader("Analyse structurelle — accords + paroles")
    st.caption(
        "L'harmonie reste le signal principal. Les répétitions de paroles "
        "renforcent les candidats de type refrain."
    )

    required = ("other", "bass", "drums")
    missing = [name for name in required if name not in stems]
    if missing:
        st.warning(
            "Stems requis manquants : " + ", ".join(missing)
        )
        return

    speech_payload = latest_cached_speech_payload(work_dir)
    words = list((speech_payload or {}).get("words", []) or [])
    if speech_payload:
        st.caption(
            "Timeline paroles utilisée : "
            f"{int(speech_payload.get('_usable_word_count', len(words)))} mots · "
            f"couverture jusqu'à {float(speech_payload.get('_coverage_end', 0.0)):.1f}s · "
            f"{Path(str(speech_payload.get('_cache_path', ''))).name}"
        )
    if not words:
        st.info(
            "Aucune analyse Whisper en cache : l'analyse harmonique reste "
            "possible, mais les refrains ne pourront pas être confirmés "
            "par les paroles."
        )

    col_meter, col_threshold = st.columns(2)
    with col_meter:
        beats_per_bar = st.selectbox(
            "Temps par mesure (prototype)",
            [2, 3, 4, 6],
            index=2,
            key="structure_beats_per_bar",
        )
    with col_threshold:
        harmonic_threshold = st.slider(
            "Seuil similarité harmonique",
            min_value=0.55,
            max_value=0.90,
            value=0.72,
            step=0.01,
            key="structure_harmonic_threshold",
        )

    analyse = st.button(
        "Analyser les motifs structurels",
        type="primary",
        width="stretch",
        key="structure_run",
    )

    cache_path = work_dir / "structure_analysis.json"
    payload = None

    if analyse:
        with st.spinner(
            "Analyse des progressions d'accords et des répétitions de paroles…"
        ):
            harmony = analyze_harmony(
                other_path=stems["other"],
                bass_path=stems["bass"],
                drums_path=stems["drums"],
                beats_per_bar=int(beats_per_bar),
            )
            harmonic = detect_harmonic_motifs(
                harmony["measures"],
                threshold=float(harmonic_threshold),
            )
            phrases = lyrics_phrases(words)
            lyric_repeats = detect_lyric_repeats(phrases)
            combined = combine_structure_candidates(
                harmonic,
                lyric_repeats,
            )
            visual_blocks = build_visual_blocks(
                harmony["measures"],
                combined,
                words,
            )

            payload = {
                "tempo": harmony["tempo"],
                "beats_per_bar": harmony["beats_per_bar"],
                "measure_count": harmony["measure_count"],
                "measures": harmony["measures"],
                "harmonic_motifs": harmonic,
                "lyric_phrases": phrases,
                "lyric_repeats": lyric_repeats,
                "candidates": combined,
                "visual_blocks": visual_blocks,
            }
            cache_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    elif cache_path.is_file():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            payload = None

    if not payload:
        return

    if not payload.get("visual_blocks"):
        payload["visual_blocks"] = build_visual_blocks(
            list(payload.get("measures", []) or []),
            list(payload.get("candidates", []) or []),
            words,
        )

    st.success(
        f"Tempo ≈ {payload.get('tempo', 0):.1f} BPM · "
        f"{payload.get('measure_count', 0)} mesures · "
        f"{len(payload.get('harmonic_motifs', []))} motifs harmoniques · "
        f"{len(payload.get('lyric_repeats', []))} répétitions de paroles."
    )

    lyric_repeats = list(payload.get("lyric_repeats", []) or [])
    if lyric_repeats:
        best_repeat = lyric_repeats[0]
        st.info(
            "Motif de paroles répété détecté · "
            f"similarité {float(best_repeat.get('similarity', 0.0)):.2f} · "
            f"{float(best_repeat.get('a_time_start', 0.0)):.1f}s ↔ "
            f"{float(best_repeat.get('b_time_start', 0.0)):.1f}s"
        )

    candidates = list(payload.get("candidates", []) or [])
    if not candidates:
        st.warning(
            "Aucun motif structurel suffisamment fort avec ces paramètres."
        )
    else:
        rows = []
        for item in candidates[:12]:
            rows.append({
                "Type": item.get("type", ""),
                "A mesures": f"{item['a_measure_start']}–{item['a_measure_end']}",
                "B mesures": f"{item['b_measure_start']}–{item['b_measure_end']}",
                "Longueur": item.get("length", 0),
                "Harmonie": item.get("harmonic_score", 0.0),
                "Paroles": item.get("lyric_score", 0.0),
                "Score combiné": item.get("combined_score", 0.0),
            })
        st.dataframe(rows, width="stretch", hide_index=True)

        best_chorus = next(
            (x for x in candidates if x.get("type") == "refrain probable"),
            None,
        )
        if best_chorus:
            st.markdown("#### Refrain probable")
            st.write(
                f"Mesures **{best_chorus['a_measure_start']}–"
                f"{best_chorus['a_measure_end']}** et **"
                f"{best_chorus['b_measure_start']}–"
                f"{best_chorus['b_measure_end']}**."
            )
            lyric_match = best_chorus.get("lyric_match") or {}
            if lyric_match:
                st.caption(
                    "Paroles répétées détectées : "
                    + str(lyric_match.get("text_a", "") or "")
                )

    st.markdown("#### Blocs + paroles")
    st.caption(
        "Prévisualisation non destructive : les blocs sont des enveloppes "
        "visuelles. Aucun timestamp de parole ou d'accord n'est déplacé."
    )

    visual_blocks = list(payload.get("visual_blocks", []) or [])
    if not visual_blocks:
        st.info("Aucun bloc visuel proposé.")
    else:
        for block in visual_blocks:
            cluster = str(block.get("cluster", "?") or "?")
            m0 = int(block.get("measure_start", 0) or 0)
            m1 = int(block.get("measure_end", 0) or 0)
            t0 = float(block.get("time_start", 0.0) or 0.0)
            t1 = float(block.get("time_end", t0) or t0)
            lyrics = str(block.get("lyrics", "") or "").strip()
            patterns = list(block.get("chord_patterns", []) or [])

            with st.container(border=True):
                title_col, time_col = st.columns([3, 1])
                with title_col:
                    st.markdown(
                        f"### Bloc {cluster} · mesures {m0}–{m1}"
                    )
                with time_col:
                    st.caption(f"{t0:.1f}s → {t1:.1f}s")

                if patterns:
                    compact_patterns = "  ·  ".join(patterns)
                    st.markdown(
                        "<div style='font-family:monospace;font-size:0.88rem;"
                        "opacity:.75;overflow-wrap:anywhere;'>"
                        + compact_patterns
                        + "</div>",
                        unsafe_allow_html=True,
                    )

                if lyrics:
                    # Readable lyric lines: sentence-ish breaks without moving timestamps.
                    lyric_lines = re.split(r"(?<=[.!?])\s+|\s{2,}", lyrics)
                    lyric_lines = [line.strip() for line in lyric_lines if line.strip()]
                    lyric_html = "<br>".join(lyric_lines)
                    st.markdown(
                        "<div style='margin-top:.65rem;padding:.75rem 1rem;"
                        "border-left:4px solid #2f80ed;"
                        "font-size:1.08rem;line-height:1.55;'>"
                        + lyric_html
                        + "</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("Section instrumentale / aucune parole détectée.")

    with st.expander("Progressions harmoniques détectées"):
        measures = list(payload.get("measures", []) or [])
        if measures:
            st.dataframe(
                [
                    {
                        "Mesure": m.get("measure"),
                        "Début": round(float(m.get("time_start", 0.0)), 2),
                        "Fin": round(float(m.get("time_end", 0.0)), 2),
                        "Accords / temps": " · ".join(m.get("beat_chords", [])),
                        "Motif": m.get("pattern", ""),
                    }
                    for m in measures
                ],
                width="stretch",
                hide_index=True,
            )

    st.download_button(
        "⬇ Exporter l'analyse structurelle JSON",
        data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="structure_analysis.json",
        mime="application/json",
        width="stretch",
    )


def main() -> None:
    st.set_page_config(
        page_title="EZScore Stem Lab",
        page_icon="🎚️",
        layout="wide",
    )

    st.title("🎚️ EZScore Stem Lab")
    st.caption(
        "Script indépendant : import audio → séparation Demucs "
        "(vocals / drums / bass / other) → lecteur multi-pistes synchronisé."
    )

    if not demucs_available():
        st.error(
            "Demucs n'est pas disponible dans cet environnement Python.\n\n"
            "Installer avec : `python -m pip install demucs`"
        )
        st.stop()

    uploaded = st.file_uploader(
        "Importer un morceau",
        type=["mp3", "wav", "ogg", "m4a", "flac", "aac", "wma"],
        accept_multiple_files=False,
    )

    if uploaded is None:
        st.info("Importe un morceau pour commencer.")
        return

    audio_hash, source = write_uploaded_source(uploaded)
    work_dir = source.parent
    stems = locate_stems(work_dir, source)

    st.write(
        f"**Fichier :** {uploaded.name}  \n"
        f"**Hash :** `{audio_hash[:12]}`  \n"
        f"**Modèle :** `{DEMUCS_MODEL}`"
    )

    if len(stems) == len(STEM_NAMES):
        st.success("Les 4 stems sont déjà présents dans le cache.")
    else:
        if st.button(
            "Extraire les 4 stems",
            type="primary",
            width="stretch",
        ):
            log_box = st.empty()
            with st.spinner(
                "Demucs sépare vocals / drums / bass / other…"
            ):
                try:
                    stems, log = extract_stems(source, work_dir)
                    log_box.code(
                        "\n".join(log.splitlines()[-25:]),
                        language="text",
                    )
                    st.success("Extraction terminée.")
                except Exception as exc:
                    st.error(str(exc))
                    st.stop()

    stems = locate_stems(work_dir, source)

    if len(stems) != len(STEM_NAMES):
        st.warning(
            "Les stems ne sont pas encore disponibles. "
            "Lance l'extraction."
        )
        return

    cached_words = latest_cached_words(work_dir)

    st.subheader("Lecteur synchronisé")
    st.caption(
        "WebAudio utilise une horloge unique pour toutes les pistes. "
        "Les paroles apparaissent dans le lecteur dès qu'une analyse Whisper "
        "a été calculée."
    )

    if not ffmpeg_available():
        st.error(
            "Les stems sont extraits, mais FFmpeg n'est pas disponible pour "
            "fabriquer les pré-écoutes MP3 légères du lecteur."
        )
        return

    render_player(
        source,
        stems,
        key=f"stem_player_{audio_hash[:12]}_{len(cached_words)}",
        words=cached_words,
    )

    st.subheader("Stems")
    download_stem_buttons(stems)

    render_speech_analysis(
        work_dir=work_dir,
        stems=stems,
    )

    render_structure_analysis(
        work_dir=work_dir,
        stems=stems,
    )

    with st.expander("Emplacements"):
        st.code(
            json.dumps(
                {
                    "source": str(source),
                    **{name: str(path) for name, path in stems.items()},
                },
                ensure_ascii=False,
                indent=2,
            ),
            language="json",
        )


if __name__ == "__main__":
    main()
