from __future__ import annotations

import json
import re
from pathlib import Path

import streamlit as st

from stemlab.config import DEMUCS_MODEL, STEM_NAMES
from stemlab.io import write_uploaded_source
from stemlab.analysis.separation import (
    demucs_available,
    extract_stems,
    locate_stems,
)
from stemlab.analysis.lyrics import (
    latest_cached_speech_payload,
    latest_cached_words,
    load_speech_cache,
    save_speech_cache,
    torch_device_label,
    transcribe_audio,
    whisper_available,
)
from stemlab.analysis.harmony import (
    analyze_harmony,
    detect_harmonic_motifs,
)
from stemlab.analysis.lyrics_structure import (
    detect_lyric_repeats,
    lyrics_phrases,
)
from stemlab.analysis.structure import (
    build_visual_blocks,
    combine_structure_candidates,
)
from stemlab.player.webaudio import (
    ffmpeg_available,
    render_player,
)


def render_speech_analysis(*, work_dir: Path, source: Path) -> None:
    st.subheader("Analyse des paroles")
    st.caption(
        "Source : audio original complet. "
        "Les stems Demucs ne sont pas utilisés pour la reconnaissance des paroles."
    )

    if not source.is_file():
        st.warning("Audio original introuvable.")
        return

    if not whisper_available():
        st.error(
            "Whisper n'est pas installé. "
            "Installer avec : `python -m pip install openai-whisper`"
        )
        return

    model_name = st.selectbox(
        "Modèle Whisper",
        ["tiny", "base", "small"],
        index=2,
        help=(
            "small est le réglage de référence utilisé par EZScore "
            "pour une meilleure reconnaissance du chant."
        ),
    )

    st.caption(f"Whisper : {torch_device_label()} · source `original`")
    cached = load_speech_cache(
        work_dir,
        model_name,
        source_name="original",
    )

    if st.button(
        "Analyser les paroles",
        type="primary",
        width="stretch",
        key=f"speech_analyse_original_{model_name}",
    ):
        with st.spinner("Whisper analyse l'audio original…"):
            payload = transcribe_audio(
                source,
                model_name=model_name,
                source_name="original",
            )
            save_speech_cache(
                work_dir,
                model_name,
                payload,
                source_name="original",
            )
            cached = payload
        st.rerun()

    if not cached:
        st.info("Clique `Analyser les paroles`.")
        return

    language = str(cached.get("language", "") or "auto")
    lyrics = str(cached.get("text", "") or "").strip()
    words = list(cached.get("words", []) or [])

    st.success(
        f"Transcription originale disponible · langue `{language}` · "
        f"{len(words)} mots."
    )

    tab_text, tab_words = st.tabs(["Paroles", "Mots horodatés"])

    with tab_text:
        st.text_area(
            "Texte transcrit",
            value=lyrics,
            height=260,
            key=f"lyrics_text_original_{model_name}",
        )

    with tab_words:
        rows = [
            {
                "Début": round(float(w.get("start", 0.0)), 3),
                "Fin": round(float(w.get("end", 0.0)), 3),
                "Mot": str(w.get("text", "") or ""),
            }
            for w in words
        ]
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
        else:
            st.info("Aucun mot horodaté.")

    export_payload = {
        "engine": str(cached.get("engine", "openai-whisper")),
        "model": str(cached.get("model", model_name)),
        "source": "original",
        "language": language,
        "text": lyrics,
        "words": words,
    }
    st.download_button(
        "⬇ Exporter l'analyse paroles JSON",
        data=json.dumps(export_payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="speech_analysis.json",
        mime="application/json",
        width="stretch",
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

    speech_payload = latest_cached_speech_payload(work_dir, source_name="original")
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
        "Architecture modulaire : séparation → timelines → analyses → "
        "structure → lecteur. Les modules sont conçus pour converger vers EZScore."
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
        if st.button("Extraire les 4 stems", type="primary", width="stretch"):
            log_box = st.empty()
            with st.spinner("Demucs sépare vocals / drums / bass / other…"):
                try:
                    stems, log = extract_stems(source, work_dir)
                    log_box.code("\n".join(log.splitlines()[-25:]), language="text")
                    st.success("Extraction terminée.")
                except Exception as exc:
                    st.error(str(exc))
                    st.stop()

    stems = locate_stems(work_dir, source)
    if len(stems) != len(STEM_NAMES):
        st.warning("Les stems ne sont pas encore disponibles. Lance l'extraction.")
        return

    cached_words = latest_cached_words(work_dir, source_name="original")

    st.subheader("Lecteur synchronisé")
    st.caption(
        "WebAudio : une horloge commune pour original + stems. "
        "Les paroles horodatées sont superposées au transport."
    )

    if not ffmpeg_available():
        st.error(
            "FFmpeg est requis pour fabriquer les pré-écoutes MP3 du lecteur."
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

    render_speech_analysis(work_dir=work_dir, source=source)
    render_structure_analysis(work_dir=work_dir, stems=stems)

    with st.expander("Architecture / emplacements"):
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
        st.code(
            "stemlab/\n"
            "  analysis/separation.py\n"
            "  analysis/lyrics.py\n"
            "  analysis/harmony.py\n"
            "  analysis/lyrics_structure.py\n"
            "  analysis/structure.py\n"
            "  models/timelines.py\n"
            "  player/webaudio.py",
            language="text",
        )


if __name__ == "__main__":
    main()
