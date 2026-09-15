from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from stemlab.config import DEMUCS_MODEL, STEM_NAMES


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
    result: dict[str, Path] = {}
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
