# Local WSL PySpark + Delta Lake Random Forest threshold tuning (path-based)
#
# Reads Delta layer created by 04_rf_train_local.py:
#   /home/arvind/ecommerce_rf/delta/rf_predictions
#
# Writes Delta layers:
#   /home/arvind/ecommerce_rf/delta/rf_threshold_metrics
#   /home/arvind/ecommerce_rf/delta/rf_tuned_predictions
#
# Purpose:
#   Evaluate the already-trained Random Forest predictions at several
#   return-probability thresholds and choose a threshold using F1 score.
#
# Run:
#   source ~/spark_env/bin/activate
#   python ~/ecommerce_rf/scripts/05_rf_predict_tune_local.py

import os

import setuptools  # Provides distutils compatibility for Python 3.12 + PySpark.
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

@F.udf(DoubleType())
def extract_return_probability(probability):
    if probability is None:
        return None

    if hasattr(probability, "asDict"):
        probability = probability.asDict(recursive=True)

    if isinstance(probability, dict):
        values = probability.get("values")
    else:
        values = getattr(probability, "values", None)

    if values is None:
        return None

    values = list(values)

    if len(values) > 1:
        return float(values[1])

    return 0.0

# -----------------------------------------------------------------------
# 1. Project paths
# -----------------------------------------------------------------------
PROJECT_DIR = "/home/arvind/ecommerce_rf"
WAREHOUSE_DIR = f"{PROJECT_DIR}/warehouse"
DELTA_DIR = f"{PROJECT_DIR}/delta"

PREDICTIONS_PATH = f"{DELTA_DIR}/rf_predictions"
THRESHOLD_METRICS_PATH = f"{DELTA_DIR}/rf_threshold_metrics"
TUNED_PREDICTIONS_PATH = f"{DELTA_DIR}/rf_tuned_predictions"

# Test the standard decision threshold plus practical alert thresholds.
THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]

# -----------------------------------------------------------------------
# 2. Validate input and create output folders
# -----------------------------------------------------------------------
if not os.path.isdir(PREDICTIONS_PATH):
    raise FileNotFoundError(
        f"Predictions Delta layer not found: {PREDICTIONS_PATH}\n"
        "Run 04_rf_train_local.py successfully first."
    )

os.makedirs(WAREHOUSE_DIR, exist_ok=True)
os.makedirs(DELTA_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# 3. Start local Spark with Delta Lake enabled
# -----------------------------------------------------------------------
builder = (
    SparkSession.builder
    .appName("EcommerceRF_ThresholdTuning_Local")
    .master("local[4]")
    .config("spark.driver.memory", "12g")
    .config("spark.driver.maxResultSize", "2g")
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
    # 4. Load durable test predictions from stage 04
    # Expected columns: returned, prediction, probability.
    # probability is a Spark Vector: [P(not returned), P(returned)].
    # -------------------------------------------------------------------
    predictions_df = spark.read.format("delta").load(PREDICTIONS_PATH)

    required_cols = {"returned", "prediction", "probability"}
    missing_cols = required_cols.difference(predictions_df.columns)
    if missing_cols:
        raise ValueError(
            "The predictions layer is missing required columns: "
            f"{sorted(missing_cols)}"
        )

    predictions_df = (
    	predictions_df
    	.withColumn("returned", F.col("returned").cast("int"))
    	.withColumn("default_prediction", F.col("prediction").cast("int"))
    	.withColumn(
        	"return_probability",
        	extract_return_probability(F.col("probability")),
    	)
    	.drop("prediction")
    )

    prediction_count = predictions_df.count()
    if prediction_count == 0:
        raise ValueError("The predictions layer contains zero rows.")

    print("Predictions input path:", PREDICTIONS_PATH)
    print("Prediction rows:", prediction_count)

    print("\nActual return-label distribution:")
    predictions_df.groupBy("returned").count().orderBy("returned").show()

    print("\nReturn-probability summary:")
    predictions_df.select(
        F.min("return_probability").alias("min_probability"),
        F.expr("percentile_approx(return_probability, 0.25)").alias("p25"),
        F.expr("percentile_approx(return_probability, 0.50)").alias("median"),
        F.expr("percentile_approx(return_probability, 0.75)").alias("p75"),
        F.max("return_probability").alias("max_probability"),
    ).show(truncate=False)

    # -------------------------------------------------------------------
    # 5. Compute binary classification metrics at each threshold
    # Positive class = returned == 1.
    # -------------------------------------------------------------------
    metric_rows = []

    for threshold in THRESHOLDS:
        scored_df = predictions_df.withColumn(
            "threshold_prediction",
            F.when(F.col("return_probability") >= F.lit(threshold), F.lit(1))
            .otherwise(F.lit(0)),
        )

        counts = scored_df.agg(
            F.sum(
                F.when(
                    (F.col("returned") == 1)
                    & (F.col("threshold_prediction") == 1),
                    1,
                ).otherwise(0)
            ).alias("tp"),
            F.sum(
                F.when(
                    (F.col("returned") == 0)
                    & (F.col("threshold_prediction") == 1),
                    1,
                ).otherwise(0)
            ).alias("fp"),
            F.sum(
                F.when(
                    (F.col("returned") == 0)
                    & (F.col("threshold_prediction") == 0),
                    1,
                ).otherwise(0)
            ).alias("tn"),
            F.sum(
                F.when(
                    (F.col("returned") == 1)
                    & (F.col("threshold_prediction") == 0),
                    1,
                ).otherwise(0)
            ).alias("fn"),
        ).first()

        tp = int(counts["tp"] or 0)
        fp = int(counts["fp"] or 0)
        tn = int(counts["tn"] or 0)
        fn = int(counts["fn"] or 0)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        accuracy = (tp + tn) / (tp + fp + tn + fn)
        specificity = tn / (tn + fp) if (tn + fp) else 0.0

        metric_rows.append(
            (
                float(threshold),
                tp,
                fp,
                tn,
                fn,
                float(precision),
                float(recall),
                float(f1),
                float(accuracy),
                float(specificity),
            )
        )

    metric_schema = [
        "threshold",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "specificity",
    ]

    metrics_df = spark.createDataFrame(metric_rows, metric_schema)

    print("\nThreshold comparison (sorted by F1):")
    metrics_df.orderBy(F.desc("f1"), F.asc("threshold")).show(
        len(THRESHOLDS), truncate=False
    )

    # Select the F1-optimal threshold. If tied, choose the lower threshold
    # so the project prioritizes catching more potentially returned orders.
    best_metric = metrics_df.orderBy(
        F.desc("f1"), F.asc("threshold")
    ).first()

    best_threshold = float(best_metric["threshold"])

    print("\nSelected threshold:", best_threshold)
    print("Selected threshold F1:", best_metric["f1"])
    print("Selected threshold precision:", best_metric["precision"])
    print("Selected threshold recall:", best_metric["recall"])

    # -------------------------------------------------------------------
    # 6. Write the threshold experiment table as a durable Delta layer
    # -------------------------------------------------------------------
    (
        metrics_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(THRESHOLD_METRICS_PATH)
    )

    # -------------------------------------------------------------------
    # 7. Write final scored rows using the selected threshold
    # -------------------------------------------------------------------
    tuned_predictions_df = (
        predictions_df
        .withColumn(
            "tuned_prediction",
            F.when(F.col("return_probability") >= F.lit(best_threshold), F.lit(1))
            .otherwise(F.lit(0)),
        )
        .withColumn("selected_threshold", F.lit(best_threshold))
        .select(
            "returned",
            "default_prediction",
            "tuned_prediction",
            "return_probability",
            "selected_threshold",
        )
    )

    (
        tuned_predictions_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(TUNED_PREDICTIONS_PATH)
    )

    # -------------------------------------------------------------------
    # 8. Validate persisted outputs
    # -------------------------------------------------------------------
    threshold_check_df = spark.read.format("delta").load(THRESHOLD_METRICS_PATH)
    tuned_check_df = spark.read.format("delta").load(TUNED_PREDICTIONS_PATH)

    print("\n===== THRESHOLD TUNING COMPLETE =====")
    print("Threshold metrics path:", THRESHOLD_METRICS_PATH)
    print("Tuned predictions path:", TUNED_PREDICTIONS_PATH)
    print("Tuned prediction rows:", tuned_check_df.count())

    print("\nStored threshold metrics:")
    threshold_check_df.orderBy(F.asc("threshold")).show(truncate=False)

    print("\nSelected-threshold confusion matrix:")
    (
        tuned_check_df
        .groupBy("returned", "tuned_prediction")
        .count()
        .orderBy("returned", "tuned_prediction")
        .show()
    )

    print("\nSample tuned predictions:")
    tuned_check_df.orderBy(F.desc("return_probability")).show(15, truncate=False)

finally:
    spark.stop()
