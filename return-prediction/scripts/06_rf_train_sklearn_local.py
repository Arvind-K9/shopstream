# Local WSL scikit-learn Random Forest training (path-based, Delta input)
#
# Reads Delta layers created by 04_rf_train_local.py:
#   /home/arvind/ecommerce_rf/delta/rf_train_split
#   /home/arvind/ecommerce_rf/delta/rf_test_split
#
# Writes local artifacts:
#   /home/arvind/ecommerce_rf/models/rf_transaction_return_sklearn.pkl
#   /home/arvind/ecommerce_rf/models/rf_transaction_return_sklearn_columns.pkl
#
# Writes MLflow run to the same local SQLite tracking store used by
# 04_rf_train_local.py, under a distinct run name for comparison.
#
# Purpose:
#   Train a scikit-learn Random Forest (with class_weight="balanced") on
#   the SAME train/test split as the Spark ML model, using one-hot encoded
#   categoricals + numeric features, so metrics are directly comparable.
#
# Run:
#   source ~/spark_env/bin/activate
#   python ~/ecommerce_rf/scripts/06_rf_train_sklearn_local.py

import os

import setuptools  # Distutils compatibility shim for Python 3.12 + PySpark.
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# -----------------------------------------------------------------------
# 1. Project paths
# -----------------------------------------------------------------------
PROJECT_DIR = "/home/arvind/ecommerce_rf"
WAREHOUSE_DIR = f"{PROJECT_DIR}/warehouse"
DELTA_DIR = f"{PROJECT_DIR}/delta"
MODELS_DIR = f"{PROJECT_DIR}/models"
MLFLOW_DB = f"{PROJECT_DIR}/mlflow.db"
MLFLOW_ARTIFACT_DIR = f"{PROJECT_DIR}/mlartifacts"

TRAIN_SPLIT_PATH = f"{DELTA_DIR}/rf_train_split"
TEST_SPLIT_PATH = f"{DELTA_DIR}/rf_test_split"

MODEL_PKL_PATH = f"{MODELS_DIR}/rf_transaction_return_sklearn.pkl"
COLUMNS_PKL_PATH = f"{MODELS_DIR}/rf_transaction_return_sklearn_columns.pkl"

EXPERIMENT_NAME = "ecommerce_rf_return_prediction"
LABEL_COL = "returned"

# -----------------------------------------------------------------------
# 2. Validate input and create output folders
# -----------------------------------------------------------------------
if not os.path.isdir(TRAIN_SPLIT_PATH) or not os.path.isdir(TEST_SPLIT_PATH):
    raise FileNotFoundError(
        "Persisted train/test split not found. Expected:\n"
        f"  {TRAIN_SPLIT_PATH}\n"
        f"  {TEST_SPLIT_PATH}\n"
        "Run 04_rf_train_local.py successfully first."
    )

os.makedirs(WAREHOUSE_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MLFLOW_ARTIFACT_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# 3. Configure local MLflow tracking (same SQLite store as stage 04)
# -----------------------------------------------------------------------
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
mlflow.set_experiment(EXPERIMENT_NAME)

# -----------------------------------------------------------------------
# 4. Start local Spark only to read Delta, then hand off to pandas
# -----------------------------------------------------------------------
builder = (
    SparkSession.builder
    .appName("EcommerceRF_SklearnTrain_Local")
    .master("local[4]")
    .config("spark.driver.memory", "10g")
    .config("spark.sql.warehouse.dir", WAREHOUSE_DIR)
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config(
        "spark.sql.catalog.spark_catalog",
        "org.apache.spark.sql.delta.catalog.DeltaCatalog",
    )
)

spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("WARN")

try:
    # -------------------------------------------------------------------
    # 5. Read the persisted Spark train/test splits and convert to pandas
    # These are the SAME rows used by the Spark ML Random Forest, so
    # metrics between the two models are directly comparable.
    # -------------------------------------------------------------------
    train_sdf = spark.read.format("delta").load(TRAIN_SPLIT_PATH)
    test_sdf = spark.read.format("delta").load(TEST_SPLIT_PATH)

    print("Train split path:", TRAIN_SPLIT_PATH)
    print("Test split path:", TEST_SPLIT_PATH)
    print("Train rows (Spark):", train_sdf.count())
    print("Test rows (Spark):", test_sdf.count())

    train_pdf = train_sdf.toPandas()
    test_pdf = test_sdf.toPandas()

    if LABEL_COL not in train_pdf.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found in train split.")

    print("\npandas train shape:", train_pdf.shape)
    print("pandas test shape:", test_pdf.shape)

finally:
    spark.stop()

# -----------------------------------------------------------------------
# 6. Separate label and build a consistent one-hot encoded feature matrix
# Categorical columns are one-hot encoded on the COMBINED train+test data
# so both splits share identical dummy columns, then split back apart.
# This avoids the "unseen category in test" column-mismatch problem.
# -----------------------------------------------------------------------
y_train = train_pdf[LABEL_COL].astype(int)
y_test = test_pdf[LABEL_COL].astype(int)

train_pdf = train_pdf.drop(columns=[LABEL_COL])
test_pdf = test_pdf.drop(columns=[LABEL_COL])

train_pdf["__split__"] = "train"
test_pdf["__split__"] = "test"

combined_pdf = pd.concat([train_pdf, test_pdf], axis=0, ignore_index=True)

categorical_cols = combined_pdf.select_dtypes(
    include=["object", "category"]
).columns.tolist()
categorical_cols = [c for c in categorical_cols if c != "__split__"]

print("\nCategorical columns one-hot encoded:", categorical_cols)

combined_encoded = pd.get_dummies(
    combined_pdf,
    columns=categorical_cols,
    dummy_na=False,
)

X_train = combined_encoded[combined_encoded["__split__"] == "train"].drop(
    columns=["__split__"]
)
X_test = combined_encoded[combined_encoded["__split__"] == "test"].drop(
    columns=["__split__"]
)

feature_columns = X_train.columns.tolist()

print("Final feature matrix shape (train):", X_train.shape)
print("Final feature matrix shape (test):", X_test.shape)
print("Total feature columns after encoding:", len(feature_columns))

print("\nTrain label distribution:")
print(y_train.value_counts())

print("\nTest label distribution:")
print(y_test.value_counts())

# -----------------------------------------------------------------------
# 7. Train scikit-learn Random Forest with class-imbalance handling
# class_weight="balanced" directly targets the low recall (~0.22) seen
# with the unweighted Spark ML Random Forest at the tuned threshold.
# -----------------------------------------------------------------------
rf_params = dict(
    n_estimators=200,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

rf = RandomForestClassifier(**rf_params)

with mlflow.start_run(run_name="rf_transaction_return_sklearn") as run:
    run_id = run.info.run_id

    for param_name, param_value in rf_params.items():
        mlflow.log_param(param_name, param_value)

    mlflow.log_param("categorical_feature_count", len(categorical_cols))
    mlflow.log_param("encoded_feature_count", len(feature_columns))
    mlflow.log_param("train_rows", int(X_train.shape[0]))
    mlflow.log_param("test_rows", int(X_test.shape[0]))
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("model_framework", "scikit-learn")

    print("\nTraining scikit-learn Random Forest (balanced class weights)...")
    rf.fit(X_train, y_train)
    print("Training completed.")

    # ---------------------------------------------------------------
    # 8. Evaluate on the held-out TEST split (not training data)
    # ---------------------------------------------------------------
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]

    auc = float(roc_auc_score(y_test, y_prob))
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    print("\nTest ROC-AUC:", auc)
    print("Test accuracy:", accuracy)
    print("Test precision:", precision)
    print("Test recall:", recall)
    print("Test F1:", f1)

    print("\nClassification report (test data):")
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(cm)

    mlflow.log_metric("test_auc", auc)
    mlflow.log_metric("test_accuracy", accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1", f1)
    mlflow.log_metric("true_negative", int(cm[0][0]))
    mlflow.log_metric("false_positive", int(cm[0][1]))
    mlflow.log_metric("false_negative", int(cm[1][0]))
    mlflow.log_metric("true_positive", int(cm[1][1]))

    # ---------------------------------------------------------------
    # 9. Log the sklearn model to MLflow for side-by-side comparison
    # ---------------------------------------------------------------
    mlflow.sklearn.log_model(rf, name="sklearn_random_forest_model")

    # ---------------------------------------------------------------
    # 10. Save local pickle artifacts (matches original task-plan output)
    # ---------------------------------------------------------------
    joblib.dump(rf, MODEL_PKL_PATH)
    joblib.dump(feature_columns, COLUMNS_PKL_PATH)

    print("\nSaved scikit-learn model to:", MODEL_PKL_PATH)
    print("Saved feature column list to:", COLUMNS_PKL_PATH)
    print("MLflow run ID:", run_id)
    print("MLflow tracking URI:", mlflow.get_tracking_uri())

print("\n===== SKLEARN RANDOM FOREST TRAINING COMPLETE =====")
print("Compare this run's metrics against the Spark ML run in the MLflow UI")
print("under experiment:", EXPERIMENT_NAME)
