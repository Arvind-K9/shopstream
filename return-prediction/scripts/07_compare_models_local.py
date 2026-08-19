# Local WSL model-comparison report for ecommerce return prediction
#
# Reads:
#   /home/arvind/ecommerce_rf/delta/rf_train_runs
#   /home/arvind/ecommerce_rf/delta/rf_threshold_metrics
#   /home/arvind/ecommerce_rf/mlflow.db
#
# Writes:
#   /home/arvind/ecommerce_rf/delta/rf_model_comparison
#   /home/arvind/ecommerce_rf/reports/rf_model_comparison.csv
#   /home/arvind/ecommerce_rf/reports/rf_model_comparison.md
#
# Run:
#   source ~/spark_env/bin/activate
#   python ~/ecommerce_rf/scripts/07_compare_models_local.py

import os
import sqlite3
from datetime import datetime, timezone

import setuptools
import pandas as pd
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

PROJECT_DIR = "/home/arvind/ecommerce_rf"
WAREHOUSE_DIR = f"{PROJECT_DIR}/warehouse"
DELTA_DIR = f"{PROJECT_DIR}/delta"
REPORTS_DIR = f"{PROJECT_DIR}/reports"
MLFLOW_DB = f"{PROJECT_DIR}/mlflow.db"

SPARK_RUNS_PATH = f"{DELTA_DIR}/rf_train_runs"
THRESHOLD_METRICS_PATH = f"{DELTA_DIR}/rf_threshold_metrics"
COMPARISON_DELTA_PATH = f"{DELTA_DIR}/rf_model_comparison"
COMPARISON_CSV_PATH = f"{REPORTS_DIR}/rf_model_comparison.csv"
COMPARISON_MD_PATH = f"{REPORTS_DIR}/rf_model_comparison.md"
EXPERIMENT_NAME = "ecommerce_rf_return_prediction"

for path in [SPARK_RUNS_PATH, THRESHOLD_METRICS_PATH]:
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Required Delta path not found: {path}")

if not os.path.isfile(MLFLOW_DB):
    raise FileNotFoundError(f"MLflow SQLite database not found: {MLFLOW_DB}")

os.makedirs(WAREHOUSE_DIR, exist_ok=True)
os.makedirs(DELTA_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def build_spark_session(app_name):
    builder = (
        SparkSession.builder
        .appName(app_name)
        .master("local[4]")
        .config("spark.driver.memory", "8g")
        .config("spark.sql.warehouse.dir", WAREHOUSE_DIR)
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()


spark = build_spark_session("EcommerceRF_ModelComparison_Read_Local")
spark.sparkContext.setLogLevel("WARN")

try:
    spark_runs_df = spark.read.format("delta").load(SPARK_RUNS_PATH)
    threshold_df = spark.read.format("delta").load(THRESHOLD_METRICS_PATH)

    latest_spark_run = spark_runs_df.orderBy(F.desc("logged_at_utc")).first()
    best_threshold = threshold_df.orderBy(F.desc("f1"), F.asc("threshold")).first()

    if latest_spark_run is None:
        raise ValueError("No Spark model run was found in rf_train_runs.")
    if best_threshold is None:
        raise ValueError("No threshold metrics were found in rf_threshold_metrics.")

    spark_row = {
        "model_name": "Spark Random Forest (tuned threshold)",
        "framework": "PySpark ML",
        "class_weighting": "None",
        "decision_threshold": float(best_threshold["threshold"]),
        "roc_auc": float(latest_spark_run["test_auc"]),
        "accuracy": float(best_threshold["accuracy"]),
        "precision": float(best_threshold["precision"]),
        "recall": float(best_threshold["recall"]),
        "f1": float(best_threshold["f1"]),
        "true_negative": int(best_threshold["true_negative"]),
        "false_positive": int(best_threshold["false_positive"]),
        "false_negative": int(best_threshold["false_negative"]),
        "true_positive": int(best_threshold["true_positive"]),
        "train_rows": int(latest_spark_run["train_rows"]),
        "test_rows": int(latest_spark_run["test_rows"]),
        "source_run_id": str(latest_spark_run["run_id"]),
        "recommendation": "High-precision return-risk alerts",
    }
finally:
    spark.stop()

conn = sqlite3.connect(MLFLOW_DB)
try:
    experiment_df = pd.read_sql_query(
        "SELECT experiment_id FROM experiments WHERE name = ?",
        conn,
        params=[EXPERIMENT_NAME],
    )
    if experiment_df.empty:
        raise ValueError(f"MLflow experiment not found: {EXPERIMENT_NAME}")

    experiment_id = str(experiment_df.iloc[0]["experiment_id"])

    sklearn_run_df = pd.read_sql_query(
        """
        SELECT r.run_uuid AS run_id
        FROM runs r
        JOIN tags t ON r.run_uuid = t.run_uuid
        WHERE r.experiment_id = ?
          AND t.key = 'mlflow.runName'
          AND t.value = 'rf_transaction_return_sklearn'
          AND r.status = 'FINISHED'
        ORDER BY r.start_time DESC
        LIMIT 1
        """,
        conn,
        params=[experiment_id],
    )

    if sklearn_run_df.empty:
        raise ValueError(
            "No completed MLflow run named rf_transaction_return_sklearn was found."
        )

    sklearn_run_id = sklearn_run_df.iloc[0]["run_id"]

    metrics_df = pd.read_sql_query(
        "SELECT key, value FROM metrics WHERE run_uuid = ?",
        conn,
        params=[sklearn_run_id],
    )
    sklearn_metrics = dict(zip(metrics_df["key"], metrics_df["value"]))

    params_df = pd.read_sql_query(
        "SELECT key, value FROM params WHERE run_uuid = ?",
        conn,
        params=[sklearn_run_id],
    )
    sklearn_params = dict(zip(params_df["key"], params_df["value"]))
finally:
    conn.close()

required_metrics = [
    "test_auc",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
]
missing_metrics = [key for key in required_metrics if key not in sklearn_metrics]
if missing_metrics:
    raise ValueError(f"Missing scikit-learn MLflow metrics: {missing_metrics}")

sklearn_row = {
    "model_name": "scikit-learn Random Forest (balanced)",
    "framework": "scikit-learn",
    "class_weighting": "balanced",
    "decision_threshold": 0.50,
    "roc_auc": float(sklearn_metrics["test_auc"]),
    "accuracy": float(sklearn_metrics["test_accuracy"]),
    "precision": float(sklearn_metrics["test_precision"]),
    "recall": float(sklearn_metrics["test_recall"]),
    "f1": float(sklearn_metrics["test_f1"]),
    "true_negative": int(sklearn_metrics["true_negative"]),
    "false_positive": int(sklearn_metrics["false_positive"]),
    "false_negative": int(sklearn_metrics["false_negative"]),
    "true_positive": int(sklearn_metrics["true_positive"]),
    "train_rows": int(sklearn_params["train_rows"]),
    "test_rows": int(sklearn_params["test_rows"]),
    "source_run_id": str(sklearn_run_id),
    "recommendation": "High-recall return detection",
}

comparison_pdf = pd.DataFrame([spark_row, sklearn_row])
generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
comparison_pdf["generated_at_utc"] = generated_at
comparison_pdf.to_csv(COMPARISON_CSV_PATH, index=False)

spark_metrics = comparison_pdf.iloc[0]
sklearn_metrics_row = comparison_pdf.iloc[1]

lines = [
    "# Ecommerce Return Prediction: Model Comparison",
    "",
    f"Generated: {generated_at}",
    "",
    "## Evaluation setup",
    "",
    f"- Training rows: {spark_row['train_rows']:,}",
    f"- Held-out test rows: {spark_row['test_rows']:,}",
    "- Positive class: `returned = 1`",
    f"- Spark Random Forest threshold: {spark_row['decision_threshold']:.2f}",
    "- scikit-learn Random Forest: `class_weight=balanced`, threshold 0.50",
    "",
    "## Metrics",
    "",
    "| Model | ROC-AUC | Accuracy | Precision | Recall | F1 | Threshold |",
    "|---|---:|---:|---:|---:|---:|---:|",
    (
        f"| Spark Random Forest (tuned) | {spark_row['roc_auc']:.4f} | "
        f"{spark_row['accuracy']:.4f} | {spark_row['precision']:.4f} | "
        f"{spark_row['recall']:.4f} | {spark_row['f1']:.4f} | "
        f"{spark_row['decision_threshold']:.2f} |"
    ),
    (
        f"| scikit-learn Random Forest (balanced) | "
        f"{sklearn_row['roc_auc']:.4f} | {sklearn_row['accuracy']:.4f} | "
        f"{sklearn_row['precision']:.4f} | {sklearn_row['recall']:.4f} | "
        f"{sklearn_row['f1']:.4f} | {sklearn_row['decision_threshold']:.2f} |"
    ),
    "",
    "## Confusion matrices",
    "",
    "### Spark Random Forest",
    "",
    "| Actual / Predicted | Not returned | Returned |",
    "|---|---:|---:|",
    f"| Not returned | {spark_row['true_negative']:,} | {spark_row['false_positive']:,} |",
    f"| Returned | {spark_row['false_negative']:,} | {spark_row['true_positive']:,} |",
    "",
    "### scikit-learn Random Forest",
    "",
    "| Actual / Predicted | Not returned | Returned |",
    "|---|---:|---:|",
    f"| Not returned | {sklearn_row['true_negative']:,} | {sklearn_row['false_positive']:,} |",
    f"| Returned | {sklearn_row['false_negative']:,} | {sklearn_row['true_positive']:,} |",
    "",
    "## Differences: scikit-learn minus Spark",
    "",
    f"- ROC-AUC: {sklearn_metrics_row['roc_auc'] - spark_metrics['roc_auc']:+.4f}",
    f"- Accuracy: {sklearn_metrics_row['accuracy'] - spark_metrics['accuracy']:+.4f}",
    f"- Precision: {sklearn_metrics_row['precision'] - spark_metrics['precision']:+.4f}",
    f"- Recall: {sklearn_metrics_row['recall'] - spark_metrics['recall']:+.4f}",
    f"- F1: {sklearn_metrics_row['f1'] - spark_metrics['f1']:+.4f}",
    "",
    "## Recommendation",
    "",
    (
        "Use the **scikit-learn balanced Random Forest** when detecting as "
        f"many likely returns as possible is the priority. It achieved recall "
        f"of {sklearn_row['recall']:.2%} and F1 of {sklearn_row['f1']:.4f}, "
        f"but produced {sklearn_row['false_positive']:,} false positives."
    ),
    "",
    (
        "Use the **Spark Random Forest at its tuned threshold** when review "
        f"capacity is limited and alerts need higher precision. It achieved "
        f"precision of {spark_row['precision']:.2%}, but recall was "
        f"{spark_row['recall']:.2%}."
    ),
    "",
    "## Reproducibility sources",
    "",
    f"- Spark MLflow run ID: `{spark_row['source_run_id']}`",
    f"- scikit-learn MLflow run ID: `{sklearn_row['source_run_id']}`",
    f"- Spark run Delta path: `{SPARK_RUNS_PATH}`",
    f"- Threshold metrics Delta path: `{THRESHOLD_METRICS_PATH}`",
    f"- MLflow database: `{MLFLOW_DB}`",
]

with open(COMPARISON_MD_PATH, "w", encoding="utf-8") as report_file:
    report_file.write("\n".join(lines) + "\n")

spark = build_spark_session("EcommerceRF_ModelComparison_Write_Local")
spark.sparkContext.setLogLevel("WARN")

try:
    comparison_sdf = spark.createDataFrame(comparison_pdf)
    (
        comparison_sdf.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(COMPARISON_DELTA_PATH)
    )

    comparison_check = spark.read.format("delta").load(COMPARISON_DELTA_PATH)

    print("\n===== MODEL COMPARISON COMPLETE =====")
    print("Comparison Delta path:", COMPARISON_DELTA_PATH)
    print("Comparison CSV report:", COMPARISON_CSV_PATH)
    print("Comparison Markdown report:", COMPARISON_MD_PATH)
    print("\nModel comparison table:")
    comparison_check.select(
        "model_name",
        "framework",
        "class_weighting",
        "decision_threshold",
        "roc_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
        "recommendation",
    ).show(truncate=False)
finally:
    spark.stop()
