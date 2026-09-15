from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
CACHE_DIR = APP_DIR / "stem_lab_data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEMUCS_MODEL = "htdemucs"
STEM_NAMES = ("vocals", "drums", "bass", "other")
SPEECH_CACHE_DIRNAME = "speech_analysis"
