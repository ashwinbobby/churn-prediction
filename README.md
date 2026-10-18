# Churn Prediction

End-to-end machine learning pipeline that predicts whether a Telco customer will churn.
The model is logistic regression implemented as a single sigmoid layer in Keras, trained
with a hyperparameter grid search tracked in MLflow, and served over a REST API with FastAPI.
This is the repo deployed on Render, that serves the frontend: https://github.com/ashwinbobby/churn-prediction-backend

## Features

- **Preprocessing** — scikit-learn `ColumnTransformer`: median imputation + standard scaling
  for numeric columns, mode imputation + one-hot encoding for categorical columns.

- **Model** — logistic regression as one `Dense(1, sigmoid)` layer with L2 regularization,
  SGD optimizer, binary cross-entropy, early stopping (patience 8) and optional balanced class
  weights.
- **Experiment tracking** — 12-run grid search logged to MLflow including params, per-epoch losses, validation metrics, cost curves, and model artifacts. Note that this information is stored on my local machine and was not pushed to the public repo.
- **Model selection** — best run chosen by validation recall → precision → AUC, then
  evaluated on the held-out test set.
- **Serving** — FastAPI inference service with a Dockerfile, deployed using Render, available at the URL above.

## Project structure

```
├── app/main.py                  FastAPI service (/health, /predict)
├── data/raw/telco_churn.csv     Input dataset
├── models/preprocessor.pkl      Fitted sklearn preprocessor (may vary from the one deployed on render)
├── src/
│   ├── preprocessing.py         Data loading + preprocessing pipeline
│   ├── train_mlflow.py          Grid search + MLflow logging (main training script)
│   └── pick_best.py             Best-run selection + test-set evaluation
└── requirements.txt
```

## Setup

Requires Python 3.11+ (developed on 3.12).

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data

The project uses the **Telco Customer Churn** dataset (IBM sample dataset, also on Kaggle):
7,043 customers, 21 columns, target `Churn` (`Yes`/`No`, ~26.5% positive class).

## Training

Run all commands from the repository root.

```bash
# Run the 12-config hyperparameter sweep
python src/train_mlflow.py

# Browse runs, params, metrics and cost curves
mlflow ui   # http://localhost:5000

# Select the best run and evaluate it on the held-out test set
python src/pick_best.py
```

**Grid searched** (12 runs): `alpha` ∈ {1e-5, 1e-4, 1e-3} × `lr` ∈ {0.01, 0.05} ×
class weights ∈ {on, off}.

**Data splits**: 80/20 stratified train/test (`random_state=42`), then 80/20 stratified
train/validation. Training runs up to 100 epochs, batch size 32, early stopping on
`val_loss` with patience 8.

## Results
**Note** - The selected model can be viewed inside the mlflow ui, it will have a tag called "selected:true".
Presented here for reference too...
Selected run `a1e-05_lr0.01_cw1` (`e209e1c5b766442b9eb9bbd72e77b1dc`) — test-set metrics:

| Metric    | Value  |
|-----------|--------|
| AUC       | 0.8398 |
| Recall    | 0.8610 |
| Precision | 0.4687 |
| F1        | 0.6070 |
| Log loss  | 0.5006 |
| Brier     | 0.1689 |

The low precision / high recall reflects the 0.4 decision threshold and the churn class
imbalance — the model is tuned to catch as many churners as possible (optimising for recall).

## API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
# or with Docker
docker build -t churn-prediction .
docker run -p 8000:8000 -e PORT=8000 churn-prediction
```

Interactive docs at `http://localhost:8000/docs`.

| Endpoint       | Description                          |
|----------------|--------------------------------------|
| `GET /health`  | `{"status": "ok"}`                   |
| `POST /predict`| Churn probability + binary prediction |
Test using the frontend at: https://churn-prediction-frontend-kappa.vercel.app/

Example request — all 21 dataset columns except `customerID` and `Churn`:

```json
POST /predict
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35,
  "TotalCharges": 844.45
}
```

Response:

```json
{
  "churn_probability": 0.7312,
  "churn_prediction": true
}
```

`churn_prediction` is `true` when `churn_probability >= 0.4`.
