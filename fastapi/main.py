from fastapi import FastAPI, HTTPException
import mlflow
import mlflow.sklearn
import numpy as np
from typing import List
import os
import logging

# ロギングの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MLflow Model Serving API")

# MLflowの設定
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5010"))
logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

# モデルのロード
model_name = "CaliforniaHousingModel"
model_stage = "Production"

try:
    logger.info(f"Attempting to load model: {model_name} from stage: {model_stage}")
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_stage}")
    logger.info(f"Successfully loaded model: {model_name} from stage: {model_stage}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

@app.get("/")
async def root():
    return {
        "message": "MLflow Model Serving API",
        "model": model_name,
        "stage": model_stage,
        "model_loaded": model is not None
    }

@app.post("/predict")
async def predict(data: List[float]):
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please check MLflow server and model deployment status."
        )
    
    try:
        # 入力データをnumpy配列に変換
        input_data = np.array(data).reshape(1, -1)
        # 予測を実行
        prediction = model.predict(input_data)
        # 予測確率を取得（可能な場合）
        try:
            probabilities = model.predict_proba(input_data)
            return {
                "prediction": prediction.tolist(),
                "probabilities": probabilities.tolist()
            }
        except:
            return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/model-info")
async def model_info():
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please check MLflow server and model deployment status."
        )
    
    try:
        return {
            "model_type": type(model).__name__,
            "model_params": model.get_params(),
            "feature_names": [
                "MedInc", "HouseAge", "AveRooms", "AveBedrms",
                "Population", "AveOccup", "Latitude", "Longitude"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 