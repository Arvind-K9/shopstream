# Local WSL PySpark + Delta Lake Random Forest training (path-based)
#
# Reads Delta layer:
#   /home/arvind/ecommerce_rf/delta/rf_transaction_features
#
# Writes Delta layers:
#   /home/arvind/ecommerce_rf/delta/rf_train_split
#   /home/arvind/ecommerce_rf/delta/rf_test_split
#   /home/arvind/ecommerce_rf/delta/rf_predictions
#   /home/arvind/ecommerce_rf/delta/rf_train_runs
#
# Writes MLflow artifacts locally:
#   /home/arvind/ecommerce_rf/mlruns
#
# Run:
#   source ~/spark_env/bin/activate
#   python ~/ecommerce_rf/scripts/04_rf_train_local.py

import gc
import os
from datetime import datetime, timezone

import setuptools
import mlflow
import mlflow.spark
from delta import configure_spark_with_delta_pip
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# -----------------------------------------------------------------------
# 1. Project paths
# -----------------------------------------------------------------------
PROJECT_DIR = "/home/arvind/ecommerce_rf"
WAREHOUSE_DIR = f"{PROJECT_DIR}/warehouse"
DELTA_DIR = f"{PROJECT_DIR}/delta"
MLFLOW_DB = f"{PROJECT_DIR}/mlflow.db"
MLFLOW_ARTIFACT_DIR = f"{PROJECT_DIR}/mlartifacts"

RF_FEATURES_PATH = f"{DELTA_DIR}/rf_transaction_features"
TRAIN_SPLIT_PATH = f"{DELTA_DIR}/rf_train_split"
TEST_SPLIT_PATH = f"{DELTA_DIR}/rf_test_split"
PREDICTIONS_PATH = f"{DELTA_DIR}/rf_predictions"
RUN_LOG_PATH = f"{DELTA_DIR}/rf_train_runs"

EXPERIMENT_NAME = "ecommerce_rf_return_prediction"

# -----------------------------------------------------------------------
# 2. Validate input and create output folders
# -----------------------------------------------------------------------
if not os.path.isdir(RF_FEATURES_PATH):
    raise FileNotFoundError(
        f"RF transaction feature layer not found: {RF_FEATURES_PATH}\n"
        "Run 03_rf_feature_prep_local.py successfully first."
    )

os.makedirs(WAREHOUSE_DIR, exist_ok=True)
os.makedirs(DELTA_DIR, exist_ok=True)
os.makedirs(MLFLOW_ARTIFACT_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# 3. Configure local MLflow tracking
# -----------------------------------------------------------------------
mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
mlflow.set_experiment(EXPERIMENT_NAME)

# -----------------------------------------------------------------------
# 4. Start local Spark with Delta Lake enabled
# -----------------------------------------------------------------------
builder = (
    SparkSession.builder
    .appName("EcommerceRF_RandomForest_Local")
    .master("local[4]")
    .config("spark.driver.memory", "16g")
    .config("spark.driver.maxResultSize", "4g")
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
    # 5. Load transaction-level model features
    # -------------------------------------------------------------------
    feature_df = spark.read.format("delta").load(RF_FEATURES_PATH)

    label_col = "returned"

    if label_col not in feature_df.columns:
        raise ValueError(f"Label column '{label_col}' was not found.")

    print("RF feature input path:", RF_FEATURES_PATH)
    print("Feature rows:", feature_df.count())
    print("Feature columns:", feature_df.columns)

    # -------------------------------------------------------------------
    # 6. Identify categorical and numerical columns
    # Lists are filtered against real columns, so the pipeline stays robust.
    # -------------------------------------------------------------------
    cat_cols = [
        "customer_gender",
        "country",
        "customer_segment",
        "product_category",
        "product_subcategory",
        "brand",
        "payment_method",
        "device_type",
        "traffic_source",
        "membership_status",
        "shipping_method",
        "warehouse_region",
        "season",
        "holiday_season",
        "day_of_week",
    ]

    num_cols = [
        "quarter",
        "customer_age",
        "unit_price",
        "quantity",
        "discount_percent",
        "discount_amount",
        "shipping_cost",
        "tax_amount",
        "order_amount",
        "delivery_days",
        "review_rating",
        "customer_lifetime_value",
        "profit_margin_percent",
        "profit_amount",
        "high_value_order",
        "order_year",
        "order_month",
        "order_day",
        "order_dow",
    ]

    cat_cols = [c for c in cat_cols if c in feature_df.columns]
    num_cols = [c for c in num_cols if c in feature_df.columns]

    if not num_cols and not cat_cols:
        raise ValueError("No usable feature columns were found.")

    print("Categorical columns:", cat_cols)
    print("Numeric columns:", num_cols)

    # -------------------------------------------------------------------
    # 7. Create a reproducible 80/20 train-test split and persist it
    # Persisting splits makes training/evaluation repeatable across scripts.
    # -------------------------------------------------------------------
    train_df, test_df = feature_df.randomSplit([0.8, 0.2], seed=42)

    (
        train_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(TRAIN_SPLIT_PATH)
    )

    (
        test_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(TEST_SPLIT_PATH)
    )

    # Reload persisted splits so subsequent stages do not depend on
    # in-memory DataFrame lineage.
    train_df = spark.read.format("delta").load(TRAIN_SPLIT_PATH)
    test_df = spark.read.format("delta").load(TEST_SPLIT_PATH)

    train_count = train_df.count()
    test_count = test_df.count()

    print("Train rows:", train_count)
    print("Test rows:", test_count)

    print("\nTrain target distribution:")
    train_df.groupBy(label_col).count().orderBy(label_col).show()

    print("\nTest target distribution:")
    test_df.groupBy(label_col).count().orderBy(label_col).show()

    # -------------------------------------------------------------------
    # 8. Build preprocessing + Random Forest pipeline
    # StringIndexer -> OneHotEncoder -> VectorAssembler -> Random Forest
    # -------------------------------------------------------------------
    stages = []

    for c in cat_cols:
        stages.append(
            StringIndexer(
                inputCol=c,
                outputCol=f"{c}_idx",
                handleInvalid="keep",
            )
        )

    if cat_cols:
        stages.append(
            OneHotEncoder(
                inputCols=[f"{c}_idx" for c in cat_cols],
                outputCols=[f"{c}_ohe" for c in cat_cols],
                handleInvalid="keep",
            )
        )

    feature_inputs = [f"{c}_ohe" for c in cat_cols] + num_cols

    stages.append(
        VectorAssembler(
            inputCols=feature_inputs,
            outputCol="features",
            handleInvalid="keep",
        )
    )

    # Local WSL has 32 GB RAM, so this can be larger than the constrained
    # Databricks serverless model. Start conservatively and tune later.
    rf = RandomForestClassifier(
        labelCol=label_col,
        featuresCol="features",
        probabilityCol="probability",
        predictionCol="prediction",
        numTrees=100,
        maxDepth=8,
        featureSubsetStrategy="sqrt",
        seed=42,
    )

    stages.append(rf)
    pipeline = Pipeline(stages=stages)

    # -------------------------------------------------------------------
    # 9. Train, evaluate, write predictions, and log model with MLflow
    # -------------------------------------------------------------------
    with mlflow.start_run(run_name="rf_transaction_return") as run:
        run_id = run.info.run_id

        mlflow.log_param("num_trees", rf.getNumTrees())
        mlflow.log_param("max_depth", rf.getMaxDepth())
        mlflow.log_param("feature_subset_strategy", rf.getFeatureSubsetStrategy())
        mlflow.log_param("categorical_feature_count", len(cat_cols))
        mlflow.log_param("numeric_feature_count", len(num_cols))
        mlflow.log_param("train_rows", train_count)
        mlflow.log_param("test_rows", test_count)
        mlflow.log_param("split_seed", 42)

        print("\nTraining Random Forest pipeline...")
        model = pipeline.fit(train_df)
        predictions = model.transform(test_df)

        evaluator_auc = BinaryClassificationEvaluator(
            labelCol=label_col,
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC",
        )

        evaluator_accuracy = MulticlassClassificationEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
            metricName="accuracy",
        )

        evaluator_f1 = MulticlassClassificationEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
            metricName="f1",
        )

        auc = float(evaluator_auc.evaluate(predictions))
        accuracy = float(evaluator_accuracy.evaluate(predictions))
        f1 = float(evaluator_f1.evaluate(predictions))

        print("RF test ROC-AUC:", auc)
        print("RF test accuracy:", accuracy)
        print("RF test F1:", f1)

        mlflow.log_metric("test_auc", auc)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1", f1)

        # Persist predictions immediately as a Delta layer.
        (
            predictions
            .select(label_col, "prediction", "probability")
            .write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .save(PREDICTIONS_PATH)
        )

        # Log the complete preprocessing + Random Forest pipeline.
        mlflow.spark.log_model(model, artifact_path="random_forest_model")

    # -------------------------------------------------------------------
    # 10. Persist model-run metrics in Delta for simple local querying
    # -------------------------------------------------------------------
    run_timestamp = datetime.now(timezone.utc).replace(tzinfo=None)

    run_log_df = spark.createDataFrame(
        [
            (
                run_id,
                EXPERIMENT_NAME,
                auc,
                accuracy,
                f1,
                int(train_count),
                int(test_count),
                int(rf.getNumTrees()),
                int(rf.getMaxDepth()),
                run_timestamp,
            )
        ],
        [
            "run_id",
            "experiment_name",
            "test_auc",
            "test_accuracy",
            "test_f1",
            "train_rows",
            "test_rows",
            "num_trees",
            "max_depth",
            "logged_at_utc",
        ],
    )

    (
        run_log_df.write
        .format("delta")
        .mode("append")
        .option("mergeSchema", "true")
        .save(RUN_LOG_PATH)
    )

    # -------------------------------------------------------------------
    # 11. Read outputs back from persistent storage for validation
    # -------------------------------------------------------------------
    predictions_check = spark.read.format("delta").load(PREDICTIONS_PATH)
    run_log_check = spark.read.format("delta").load(RUN_LOG_PATH)

    print("\n===== RANDOM FOREST TRAINING COMPLETE =====")
    print("Predictions path:", PREDICTIONS_PATH)
    print("Prediction rows:", predictions_check.count())
    print("Run log path:", RUN_LOG_PATH)
    print("MLflow run ID:", run_id)
    print("MLflow tracking URI:", mlflow.get_tracking_uri())

    print("\nPrediction sample:")
    predictions_check.show(10, truncate=False)

    print("\nConfusion-matrix-style counts:")
    (
        predictions_check
        .groupBy(label_col, "prediction")
        .count()
        .orderBy(label_col, "prediction")
        .show()
    )

    print("\nLatest stored run metrics:")
    run_log_check.orderBy(F.desc("logged_at_utc")).show(5, truncate=False)

    # Helpful feature-importance output for later model interpretation.
    rf_model = model.stages[-1]
    importances = rf_model.featureImportances.toArray().tolist()
    print("\nRandom Forest feature vector size:", len(importances))

    # Local Python memory cleanup after all durable writes are complete.
    del model, predictions, pipeline, train_df, test_df, feature_df
    gc.collect()

finally:
    spark.stop()
