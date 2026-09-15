import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
)
import joblib
from preprocessing import load_data, build_preprocessor

mlflow.set_experiment("churn-prediction-tf")
mlflow.tensorflow.autolog()

df = load_data("data/raw/telco_churn.csv")
X = df.drop(columns=["Churn"])
y = df["Churn"].values

#the cross-vaidation set will be created during fitting
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y,test_size=0.2, stratify=y, random_state=42)
preprocessor , num_cols, cat_cols = build_preprocessor(df)
X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

if(hasattr(X_train, "toarray")):
    X_train = X_train.toarray()
    X_test = X_test.toarray()

joblib.dump(preprocessor, "models/preprocessor.pkl")

neg, pos = np.bincount(y_train)
total = neg + pos
class_weights = {0: total/(2*neg), 1: total/(2*pos)}

def build_model(input_dim):
    # model = keras.Sequential([
    #     keras.layers.Input(shape=(input_dim,)),
    #     keras.layers.Dense(64, activation="relu"),
    #     keras.layers.Dense(32, activation="relu"),
    #     keras.layers.Dense(1, activation="sigmoid"),
    # ])
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model

with mlflow.start_run(run_name="keras_baseline"):
    model = build_model(X_train.shape[1])

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=8, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        class_weight=class_weights,
        callbacks=[early_stop],
        verbose=2,
    )

    probs = model.predict(X_test).ravel()
    preds = (probs>=0.5).astype(int)

    metrics = {
        "test_accuracy": accuracy_score(y_test, preds),
        "test_precision": precision_score(y_test, preds),
        "test_recall": recall_score(y_test, preds),
        "test_f1": f1_score(y_test, preds),
        "test_roc_auc": roc_auc_score(y_test, probs),
    }
    mlflow.log_metrics(metrics)
    print(metrics)
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))

    model.save("models/churn_model.keras")


