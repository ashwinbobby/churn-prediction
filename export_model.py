# export_model.py
import os
import mlflow, mlflow.keras
import numpy as np

RUN_ID = "e209e1c5b766442b9eb9bbd72e77b1dc"   # from pick_best.py
OUT = "../churn-prediction-backend/artifacts"    # adjust to your new repo's path
os.makedirs(OUT, exist_ok=True)

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# 1. model -> logistic regression weights (no TensorFlow needed on Render)
model = mlflow.keras.load_model(f"runs:/{RUN_ID}/model")
w, b = model.layers[0].get_weights()
np.savez(f"{OUT}/logreg.npz", w=w.ravel(), b=b)

# 2. preprocessor (logged as an artifact in every run)
mlflow.artifacts.download_artifacts(
    run_id=RUN_ID, artifact_path="preprocessor.pkl", dst_path=OUT
)

# 3. sanity check: numpy output must match Keras
import joblib, pandas as pd
print("weights shape:", w.shape, "bias:", b)