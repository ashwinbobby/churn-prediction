import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend, safe for scripts/servers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, log_loss,
)
import joblib
from preprocessing import load_data, build_preprocessor

df = load_data("data/raw/telco_churn.csv")
X = df.drop(columns=["Churn"])
y = df["Churn"].values

X_train_raw, X_test_raw, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
preprocessor, num_cols, cat_cols = build_preprocessor(df)
X_train_full = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

if hasattr(X_train_full, "toarray"):
    X_train_full = X_train_full.toarray()
    X_test = X_test.toarray()

# carve out a cross-validation set from the training data (mirrors the
# validation_split=0.2 the Keras version used internally)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, stratify=y_train_full, random_state=42
)

joblib.dump(preprocessor, "models/preprocessor.pkl")

neg, pos = np.bincount(y_train)
total = neg + pos
class_weight = {0: total / (2 * neg), 1: total / (2 * pos)}


def plot_cost_curves(train_costs, val_costs, out_path="models/cost_curve.png"):
    """Plot training vs. validation loss (cost) per epoch and save to disk."""
    epochs = range(1, len(train_costs) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_costs, label="Training cost", marker="o", markersize=3)
    plt.plot(epochs, val_costs, label="Cross-validation cost", marker="o", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Cost (log loss)")
    plt.title("Training vs. Cross-Validation Cost")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


# Logistic regression trained via SGD so we can track cost per epoch.
# loss="log_loss" + partial_fit gives us the same "cost curve over epochs"
# view we had for the neural net, but for a plain linear model.
model = SGDClassifier(
    loss="log_loss",
    penalty="l2",
    alpha=1e-4,          # L2 regularization strength
    learning_rate="optimal",
    random_state=42,
)

n_epochs = 100
patience = 8
best_val_loss = np.inf
best_epoch = 0
best_coef, best_intercept = None, None
epochs_no_improve = 0

train_costs, val_costs = [], []
classes = np.unique(y_train)
rng = np.random.default_rng(42)

for epoch in range(1, n_epochs + 1):
    # shuffle each epoch, like Keras does by default
    perm = rng.permutation(len(X_train))
    X_train_shuffled, y_train_shuffled = X_train[perm], y_train[perm]

    sample_weight = np.array([class_weight[label] for label in y_train_shuffled])
    model.partial_fit(X_train_shuffled, y_train_shuffled, classes=classes, sample_weight=sample_weight)

    train_probs = model.predict_proba(X_train)[:, 1]
    val_probs = model.predict_proba(X_val)[:, 1]
    train_loss = log_loss(y_train, train_probs)
    val_loss = log_loss(y_val, val_probs)

    train_costs.append(train_loss)
    val_costs.append(val_loss)
    print(f"epoch {epoch:3d}/{n_epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_coef = model.coef_.copy()
        best_intercept = model.intercept_.copy()
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (best epoch: {best_epoch})")
            break

# restore best weights (mirrors restore_best_weights=True from the Keras version)
model.coef_ = best_coef
model.intercept_ = best_intercept

# --- Cost curve plot (train vs. cross-validation) ---
cost_curve_path = plot_cost_curves(train_costs, val_costs)

total_epochs_run = len(train_costs)

probs = model.predict_proba(X_test)[:, 1]
preds = (probs >= 0.5).astype(int)

metrics = {
    "test_accuracy": accuracy_score(y_test, preds),
    "test_precision": precision_score(y_test, preds),
    "test_recall": recall_score(y_test, preds),
    "test_f1": f1_score(y_test, preds),
    "test_roc_auc": roc_auc_score(y_test, probs),
}

run_info = {
    "model": "logistic_regression (SGDClassifier, log_loss)",
    "epochs_run": total_epochs_run,
    "best_epoch": best_epoch,
    "best_val_loss": best_val_loss,
    "restore_best_weights": True,
    "metrics": metrics,
}
with open("models/run_info.json", "w") as f:
    json.dump(run_info, f, indent=2)

print(metrics)
print(f"Model saved corresponds to epoch {best_epoch} of {total_epochs_run} run "
      f"(best val_loss, restore_best_weights=True)")
print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))

joblib.dump(model, "models/churn_model.pkl")
print(f"Cost curve plot saved to {cost_curve_path}")
print("Run info saved to models/run_info.json")