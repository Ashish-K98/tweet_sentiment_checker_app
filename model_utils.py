from config import MODEL_PATH
import joblib
from typing import List

_model = None

def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def predict(texts: List[str]):
    model = get_model()
    preds = model.predict(texts)
    probs = model.predict_proba(texts)
    return preds.tolist(), probs.tolist()

    