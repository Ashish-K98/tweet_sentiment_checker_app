from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from model_utils import predict
import uvicorn
import time
import logging

app = FastAPI(title="Sentiment API")

logging.basicConfig(level=logging.INFO)

class PredictRequest(BaseModel):
    texts: List[str]

class PredictResponse(BaseModel):
    predictions: List[int]
    probabilities: List[List[float]]
    latency_ms: Optional[float]

@app.get("/")
def read_root():
    return {"message": "Welcome! FastAPI is running."}

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(req: PredictRequest):
    start = time.time()
    preds, probs = predict(req.texts)
    # if preds==0:
    #     preds=['Negative']
    # elif preds==1:
    #     preds=['Neutral']
    # else:
    #     preds=['Positive']    
         
    latency = (time.time() - start) * 1000
    logging.info(f"pred_count={len(req.texts)} latency_ms={latency:.2f}")
    return {"predictions": preds, "probabilities": probs, "latency_ms": latency}

# if __name__ == "__main__":
    # uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)