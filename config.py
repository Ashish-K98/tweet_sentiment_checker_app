from pathlib import Path

ROOT = Path(__file__).resolve().parents[0]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "sentiment_tfidf_lr_v1.joblib"

RANDOM_SEED = 42