from __future__ import annotations

import base64
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import streamlit as st

from stemlab.config import STEM_NAMES

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


