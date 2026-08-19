# Local WSL PySpark + Delta Lake RF feature preparation (path-based)
#
# Reads Delta layer:
#   /home/arvind/ecommerce_rf/delta/silver_orders
#
# Writes Delta layer:
#   /home/arvind/ecommerce_rf/delta/rf_transaction_features
#
# Purpose:
#   Build a transaction-level ML feature dataset for predicting Returned.
#   Each output row represents one order/transaction.
#
# Run:
#   source ~/spark_env/bin/activate
#   python ~/ecommerce_rf/scripts/03_rf_feature_prep_local.py

import os

import setuptools
from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# -----------------------------------------------------------------------
# 1. Project paths
# -----------------------------------------------------------------------
PROJECT_DIR = "/home/arvind/ecommerce_rf"
WAREHOUSE_DIR = f"{PROJECT_DIR}/warehouse"
DELTA_DIR = f"{PROJECT_DIR}/delta"

SILVER_PATH = f"{DELTA_DIR}/silver_orders"
RF_FEATURES_PATH = f"{DELTA_DIR}/rf_transaction_features"

# -----------------------------------------------------------------------
# 2. Validate input path and create output folders
# -----------------------------------------------------------------------
if not os.path.isdir(SILVER_PATH):
    raise FileNotFoundError(
        f"Silver Delta layer not found: {SILVER_PATH}\n"
        "Run 01_etl_local.py successfully before this script."
    )

os.makedirs(WAREHOUSE_DIR, exist_ok=True)
os.makedirs(DELTA_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# 3. Start local Spark with Delta Lake enabled
# -----------------------------------------------------------------------
builder = (
    SparkSession.builder
    .appName("EcommerceRF_RF_FeaturePrep_Local")
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
    # 4. Read transaction-level Silver data
    # -------------------------------------------------------------------
    df = spark.read.format("delta").load(SILVER_PATH)

    print("Silver input path:", SILVER_PATH)
    print("Input row count:", df.count())

    # Convert Silver field names to lowercase for all ML stages.
    df = df.toDF(*[c.lower() for c in df.columns])
    print("Columns after lowercasing:", df.columns)

    # -------------------------------------------------------------------
    # 5. Derive date features
    # -------------------------------------------------------------------
    if "order_date" in df.columns:
        df = (
            df
            .withColumn("order_year", F.year("order_date"))
            .withColumn("order_month", F.month("order_date"))
            .withColumn("order_day", F.dayofmonth("order_date"))
            .withColumn("order_dow", F.dayofweek("order_date"))
            .drop("order_date")
        )

    # -------------------------------------------------------------------
    # 6. Remove identifiers and known leakage fields
    # IDs identify a row/entity but do not generalize to future orders.
    # order_status is excluded because it can be known after fulfillment.
    # membership_status_ord duplicates membership_status categorically.
    # -------------------------------------------------------------------
    drop_cols = [
        c for c in [
            "order_id",
            "customer_id",
            "product_id",
            "order_status",
            "membership_status_ord",
        ]
        if c in df.columns
    ]

    if drop_cols:
        df = df.drop(*drop_cols)

    print("Dropped ID/leakage columns:", drop_cols)

    # -------------------------------------------------------------------
    # 7. Define label, numerical features, and categorical features
    # -------------------------------------------------------------------
    if "returned" not in df.columns:
        raise ValueError("The label column 'returned' is missing from Silver data.")

    df = df.withColumn("returned", F.col("returned").cast("int"))

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

    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]

    print("Numeric feature columns:", num_cols)
    print("Categorical feature columns:", cat_cols)

    for c in num_cols:
        df = df.withColumn(c, F.col(c).cast("double"))

    for c in cat_cols:
        df = df.withColumn(c, F.col(c).cast("string"))

    # -------------------------------------------------------------------
    # 8. Frequency-cap high-cardinality categorical columns
    #
    # This reduces sparse one-hot columns in training while avoiding
    # target leakage. Categories with fewer than MIN_FREQ_COUNT rows are
    # combined into the shared "Other" category.
    # -------------------------------------------------------------------
    HIGH_CARD_COLS = ["brand", "product_subcategory"]
    MIN_FREQ_COUNT = 30

    def cap_rare_categories(input_df, column_name, min_count):
        frequency_df = (
            input_df.groupBy(column_name)
            .count()
            .withColumnRenamed("count", "category_frequency")
        )

        return (
            input_df.join(frequency_df, on=column_name, how="left")
            .withColumn(
                column_name,
                F.when(F.col("category_frequency") < min_count, "Other")
                .otherwise(F.col(column_name)),
            )
            .drop("category_frequency")
        )

    for c in HIGH_CARD_COLS:
        if c in df.columns:
            distinct_before = df.select(c).distinct().count()
            df = cap_rare_categories(df, c, MIN_FREQ_COUNT)
            distinct_after = df.select(c).distinct().count()
            print(
                f"{c}: distinct categories reduced from "
                f"{distinct_before} to {distinct_after}; "
                f"minimum frequency = {MIN_FREQ_COUNT}"
            )

    # -------------------------------------------------------------------
    # 9. Select the model feature layer and remove incomplete rows
    # -------------------------------------------------------------------
    feature_cols = ["returned"] + cat_cols + num_cols
    feature_df = df.select(*feature_cols).dropna()

    print("Rows after selecting features and dropping nulls:", feature_df.count())

    # -------------------------------------------------------------------
    # 10. Validate class balance before writing the feature layer
    # -------------------------------------------------------------------
    print("\nReturned label distribution:")
    feature_df.groupBy("returned").count().orderBy("returned").show()

    # -------------------------------------------------------------------
    # 11. Write feature data to a persistent Delta path
    # -------------------------------------------------------------------
    (
        feature_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(RF_FEATURES_PATH)
    )

    # -------------------------------------------------------------------
    # 12. Final validation: read back from disk
    # -------------------------------------------------------------------
    rf_features_df = spark.read.format("delta").load(RF_FEATURES_PATH)

    print("\n===== RF FEATURE PREPARATION COMPLETE =====")
    print("RF feature Delta path:", RF_FEATURES_PATH)
    print("Final transaction rows:", rf_features_df.count())
    print("Final column count:", len(rf_features_df.columns))
    print("Final columns:", rf_features_df.columns)

    print("\nSample transaction features:")
    rf_features_df.show(10, truncate=False)

    print("\nTop category values after frequency capping:")
    for c in HIGH_CARD_COLS:
        if c in rf_features_df.columns:
            print(f"\n{c}:")
            rf_features_df.groupBy(c).count().orderBy(F.desc("count")).show(
                20, truncate=False
            )

finally:
    spark.stop()
