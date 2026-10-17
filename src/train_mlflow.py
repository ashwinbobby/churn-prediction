import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss,
    precision_score, recall_score, f1_score,
)
from preprocessing import load_data, build_preprocessor

# ---------- MLflow setup ----------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("telco-churn")

# ---------- data (done once) ----------
df = load_data("data/raw/telco_churn.csv")
X, y = df.drop(columns=["Churn"]), df["Churn"].values

X_train_raw, X_test_raw, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
preprocessor = build_preprocessor(df)
X_train_full = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)
if hasattr(X_train_full, "toarray"):
    X_train_full, X_test = X_train_full.toarray(), X_test.toarray()

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
)
joblib.dump(preprocessor, "models/preprocessor.pkl")

neg, pos = np.bincount(y_train)
total = neg + pos
balanced_weights = {0: total / (2 * neg), 1: total / (2 * pos)}

MAX_EPOCHS, BATCH_SIZE, PATIENCE = 100, 32, 8


# ---------- logistic regression = one sigmoid unit + L2 + log loss ----------
def build_model(n_features, alpha, lr):
    model = keras.Sequential([
        keras.layers.Input(shape=(n_features,)),
        keras.layers.Dense(1, activation="sigmoid",
                           kernel_regularizer=keras.regularizers.l2(alpha / 2)),
    ])
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr),
                  loss="binary_crossentropy")
    return model


# ---------- one tracked run ----------
def run_experiment(params):
    keras.backend.clear_session()
    tf.keras.utils.set_random_seed(42)

    run_name = f"a{params['alpha']}_lr{params['lr']}_cw{int(params['use_class_weight'])}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_params({"max_epochs": MAX_EPOCHS, "batch_size": BATCH_SIZE,
                           "patience": PATIENCE})

        model = build_model(X_train.shape[1], params["alpha"], params["lr"])

        # per-epoch curves in the MLflow UI
        log_epochs = keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: mlflow.log_metrics(
                {"epoch_train_loss": logs["loss"],
                 "epoch_val_loss": logs["val_loss"]}, step=epoch))

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, shuffle=True,
            class_weight=balanced_weights if params["use_class_weight"] else None,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=PATIENCE, restore_best_weights=True),
                log_epochs,
            ],
            verbose=0,
        )

        train_costs = history.history["loss"]
        val_costs = history.history["val_loss"]
        epochs_run = len(train_costs)
        best_epoch = int(np.argmin(val_costs)) + 1

        # validation metrics (used to select the winner).
        # Computed with sklearn so they exclude the L2 penalty and are
        # comparable across different alpha values.
        val_probs = model.predict(X_val, verbose=0).ravel()
        val_preds = (val_probs >= 0.4).astype(int)
        mlflow.log_metrics({
            "val_log_loss": log_loss(y_val, val_probs),
            "val_brier": brier_score_loss(y_val, val_probs),
            "val_auc": roc_auc_score(y_val, val_probs),
            "val_precision": precision_score(y_val, val_preds),
            "val_recall": recall_score(y_val, val_preds),
            "val_f1": f1_score(y_val, val_preds),
            "epochs_run": epochs_run,
            "best_epoch": best_epoch,
        })

        # cost curve
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, epochs_run + 1), train_costs, label="Training cost")
        ax.plot(range(1, epochs_run + 1), val_costs, label="Validation cost")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Cost")
        ax.legend(); ax.grid(True, alpha=0.3)
        mlflow.log_figure(fig, "cost_curve.png")
        plt.close(fig)

        mlflow.log_artifact("models/preprocessor.pkl")
        mlflow.keras.log_model(model, name="model")

        print(f"{run_name}: val_log_loss={log_loss(y_val, val_probs):.4f} "
              f"val_auc={roc_auc_score(y_val, val_probs):.4f}")


# ---------- the sweep ----------
grid = {
    "alpha": [1e-5, 1e-4, 1e-3],
    "lr": [0.01, 0.05],
    "use_class_weight": [True, False],
}

if __name__ == "__main__":
    for values in itertools.product(*grid.values()):
        run_experiment(dict(zip(grid.keys(), values)))