import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# Build logistic regression model
model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1],)),
    layers.Dense(
        1,
        activation="sigmoid",
        kernel_regularizer=regularizers.l2(1e-4)
    )
])

model.compile(
    optimizer=keras.optimizers.SGD(),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Automatically stop and restore the best weights
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    class_weight=class_weight,
    callbacks=[early_stopping],
    verbose=1
)

# Plot training and validation loss
plt.plot(history.history["loss"], label="Training cost")
plt.plot(history.history["val_loss"], label="Validation cost")
plt.xlabel("Epoch")
plt.ylabel("Cost (binary cross-entropy)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("models/cost_curve.png", dpi=150)
plt.close()

# Evaluate on test data
probs = model.predict(X_test, verbose=0).ravel()
preds = (probs >= 0.4).astype(int)

metrics = {
    "test_accuracy": accuracy_score(y_test, preds),
    "test_precision": precision_score(y_test, preds),
    "test_recall": recall_score(y_test, preds),
    "test_f1": f1_score(y_test, preds),
    "test_roc_auc": roc_auc_score(y_test, probs),
}

print(metrics)
print(classification_report(y_test, preds))
print(confusion_matrix(y_test, preds))

model.save("models/churn_model.keras")