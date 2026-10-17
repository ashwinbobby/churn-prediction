import joblib
import mlflow
import mlflow.keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, log_loss, brier_score_loss,
                             precision_score, recall_score, f1_score)
from preprocessing import load_data

mlflow.set_tracking_uri("sqlite:///mlflow.db")

runs = mlflow.search_runs(experiment_names=["telco-churn"],
                          order_by=["metrics.val_log_loss ASC"])
cols = ["run_id", "params.alpha", "params.lr", "params.use_class_weight",
        "metrics.val_log_loss", "metrics.val_auc"]
print(runs[cols].head())
best_run_id = runs.iloc[0]["run_id"]

df = load_data("data/raw/telco_churn.csv")
X, y = df.drop(columns=["Churn"]), df["Churn"].values
_, X_test_raw, _, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

preprocessor = joblib.load("models/preprocessor.pkl")
X_test = preprocessor.transform(X_test_raw)
if hasattr(X_test, "toarray"):
    X_test = X_test.toarray()

model = mlflow.keras.load_model(f"runs:/{best_run_id}/model")
probs = model.predict(X_test, verbose=0).ravel()
preds = (probs >= 0.4).astype(int)

test_metrics = {
    "test_log_loss": log_loss(y_test, probs),
    "test_brier": brier_score_loss(y_test, probs),
    "test_auc": roc_auc_score(y_test, probs),
    "test_precision": precision_score(y_test, preds),
    "test_recall": recall_score(y_test, preds),
    "test_f1": f1_score(y_test, preds),
}
print(test_metrics)

with mlflow.start_run(run_id=best_run_id):
    mlflow.log_metrics(test_metrics)
    mlflow.set_tag("selected", "true")