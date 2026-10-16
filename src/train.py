import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)
from preprocessing import load_data, build_preprocessor

tf.keras.utils.set_random_seed(42)

# --- data ---
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
class_weight = {0: total / (2 * neg), 1: total / (2 * pos)}

# --- model: logistic regression = one sigmoid unit + L2 + log loss ---
alpha = 1e-4
model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(1, activation="sigmoid",
                       kernel_regularizer=keras.regularizers.l2(alpha / 2)),
])
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.05),
              loss="binary_crossentropy")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100, batch_size=32, shuffle=True,
    class_weight=class_weight,
    callbacks=[keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True)],
    verbose=2,
)

train_costs, val_costs = history.history["loss"], history.history["val_loss"]
epochs_run = len(train_costs)
best_epoch = int(np.argmin(val_costs)) + 1

# --- cost curve ---
plt.figure(figsize=(8, 5))
plt.plot(range(1, epochs_run + 1), train_costs, label="Training cost", marker="o", markersize=3)
plt.plot(range(1, epochs_run + 1), val_costs, label="Cross-validation cost", marker="o", markersize=3)
plt.xlabel("Epoch"); plt.ylabel("Cost (log loss)")
plt.title("Training vs. Cross-Validation Cost")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("models/cost_curve.png", dpi=150); plt.close()

# --- evaluate ---
probs = model.predict(X_test, verbose=0).ravel()
preds = (probs >= 0.4).astype(int)

metrics = {
    "test_accuracy": accuracy_score(y_test, preds),
    "test_precision": precision_score(y_test, preds),
    "test_recall": recall_score(y_test, preds),
    "test_f1": f1_score(y_test, preds),
    "test_roc_auc": roc_auc_score(y_test, probs),
}
with open("models/run_info.json", "w") as f:
    json.dump({
        "model": "logistic_regression (Keras Dense(1, sigmoid), L2, SGD)",
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "best_val_loss": float(min(val_costs)),
        "restore_best_weights": True,
        "metrics": metrics,
    }, f, indent=2)

print(metrics)
print(f"Model saved corresponds to epoch {best_epoch} of {epochs_run} run")
print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))

model.save("models/churn_model.keras")