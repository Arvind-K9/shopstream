# Local WSL PySpark + Delta Lake ETL (path-based version)
#
# Reads:
#   /home/arvind/ecommerce_rf/raw/ecommerce_orders_dataset.csv
#
# Writes Delta Lake data to:
#   /home/arvind/ecommerce_rf/delta/silver_orders
#
# This script deliberately uses Delta filesystem paths, not Spark SQL table
# names. That makes the Delta data available to separate local Spark sessions,
# scripts, and future Airflow tasks without configuring a Hive metastore.
#
# Run:
#   source ~/spark_env/bin/activate
#   python ~/ecommerce_rf/scripts/01_etl_local.py

import os

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    concat_ws,
    create_map,
    lit,
    lower,
    regexp_replace,
    sum as spark_sum,
    to_date,
    trim,
    when,
)
from pyspark.sql.types import DoubleType, IntegerType, StringType

# -----------------------------------------------------------------------
# 1. Project paths
# -----------------------------------------------------------------------
PROJECT_DIR = "/home/arvind/ecommerce_rf"
RAW_FILE = f"{PROJECT_DIR}/raw/ecommerce_orders_dataset.csv"
WAREHOUSE_DIR = f"{PROJECT_DIR}/warehouse"
DELTA_DIR = f"{PROJECT_DIR}/delta"
SILVER_PATH = f"{DELTA_DIR}/silver_orders"

# -----------------------------------------------------------------------
# 2. Confirm input and create output folders
# -----------------------------------------------------------------------
if not os.path.isfile(RAW_FILE):
    raise FileNotFoundError(
        f"Raw CSV not found: {RAW_FILE}\n"
        "Check the exact filename with: ls -lh ~/ecommerce_rf/raw/"
    )

os.makedirs(WAREHOUSE_DIR, exist_ok=True)
os.makedirs(DELTA_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# 3. Start local Spark with Delta Lake enabled
# -----------------------------------------------------------------------
builder = (
    SparkSession.builder
    .appName("EcommerceRF_ETL_Local")
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
    # 4. Read raw CSV
    # -------------------------------------------------------------------
    print(f"Reading raw CSV: {RAW_FILE}")

    df = (
        spark.read
        .option("header", True)
        .option("escape", '"')
        .option("multiLine", False)
        .csv(RAW_FILE)
    )

    print("Raw row count:", df.count())
    print("Raw columns:", df.columns)

    # -------------------------------------------------------------------
    # 5. Standardize column names
    # -------------------------------------------------------------------
    standardized_cols = [
        name.strip()
        .replace(" ", "_")
        .replace(".", "")
        .replace("/", "_")
        .replace("-", "_")
        for name in df.columns
    ]

    df = df.toDF(*standardized_cols)

    # -------------------------------------------------------------------
    # 6. Clean whitespace in all raw string columns
    # -------------------------------------------------------------------
    for field in df.schema.fields:
        if isinstance(field.dataType, StringType):
            df = df.withColumn(
                field.name,
                regexp_replace(trim(col(field.name)), r"\s+", " "),
            )

    # -------------------------------------------------------------------
    # 7. Create Order_Date from Month, Day, Year and remove source fields
    # -------------------------------------------------------------------
    date_parts = ["Month", "Day", "Year"]

    if all(c in df.columns for c in date_parts):
        df = df.withColumn(
            "Order_Date",
            to_date(
                concat_ws(
                    "/",
                    col("Month").cast("string"),
                    col("Day").cast("string"),
                    col("Year").cast("string"),
                ),
                "M/d/yyyy",
            ),
        )
        df = df.drop(*date_parts)

    # -------------------------------------------------------------------
    # 8. Cast numeric columns
    # -------------------------------------------------------------------
    numeric_cols_double = [
        "Unit_Price",
        "Discount_Amount",
        "Shipping_Cost",
        "Tax_Amount",
        "Order_Amount",
        "Review_Rating",
        "Customer_Lifetime_Value",
        "Profit_Amount",
        "Profit_Margin_Percent",
    ]

    numeric_cols_int = [
        "Quantity",
        "Discount_Percent",
        "Delivery_Days",
        "Quarter",
        "Customer_Age",
    ]

    for column_name in numeric_cols_double:
        if column_name in df.columns:
            df = df.withColumn(column_name, col(column_name).cast(DoubleType()))

    for column_name in numeric_cols_int:
        if column_name in df.columns:
            df = df.withColumn(column_name, col(column_name).cast(IntegerType()))

    # -------------------------------------------------------------------
    # 9. Convert binary categorical fields into 0/1 integer fields
    # -------------------------------------------------------------------
    binary_cols = ["Coupon_Used", "Returned", "High_Value_Order"]

    for column_name in binary_cols:
        if column_name in df.columns:
            df = df.withColumn(
                column_name,
                when(col(column_name).rlike("(?i)^(yes|y|true|1)$"), 1)
                .otherwise(0)
                .cast(IntegerType()),
            )

    # -------------------------------------------------------------------
    # 10. Normalize categorical text to lowercase
    # -------------------------------------------------------------------
    lower_cols = [
        "Customer_Gender",
        "Payment_Method",
        "Device_Type",
        "Traffic_Source",
        "Order_Status",
        "Membership_Status",
        "Season",
        "Holiday_Season",
        "Shipping_Method",
        "Warehouse_Region",
        "Customer_Segment",
        "Country",
        "City",
        "Brand",
        "Product_Category",
        "Product_Subcategory",
    ]

    for column_name in lower_cols:
        if column_name in df.columns:
            df = df.withColumn(column_name, lower(col(column_name)))

    # -------------------------------------------------------------------
    # 11. Membership ordinal helper feature
    # -------------------------------------------------------------------
    if "Membership_Status" in df.columns:
        membership_map = {
            "standard": 0,
            "silver": 1,
            "gold": 2,
            "platinum": 3,
        }

        membership_expr = create_map(
            [item for pair in membership_map.items() for item in (lit(pair[0]), lit(pair[1]))]
        )

        df = df.withColumn(
            "Membership_Status_ord",
            membership_expr[col("Membership_Status")],
        )

    # -------------------------------------------------------------------
    # 12. Data-quality checks
    # -------------------------------------------------------------------
    print("\nNull counts after cleaning:")
    null_counts = df.select(
        [
            spark_sum(when(col(column_name).isNull(), 1).otherwise(0)).alias(
                column_name
            )
            for column_name in df.columns
        ]
    )
    null_counts.show(truncate=False)

    if "Order_Date" in df.columns:
        null_order_dates = df.select(
            spark_sum(when(col("Order_Date").isNull(), 1).otherwise(0)).alias(
                "null_order_dates"
            )
        ).first()["null_order_dates"]
        print("Null Order_Date rows:", null_order_dates)

    # -------------------------------------------------------------------
    # 13. Write the cleaned Silver Delta layer to a filesystem path
    # -------------------------------------------------------------------
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(SILVER_PATH)
    )

    print(f"\nSilver Delta layer written successfully: {SILVER_PATH}")

    # -------------------------------------------------------------------
    # 14. Read back from disk to prove the independent path is valid
    # -------------------------------------------------------------------
    silver_df = spark.read.format("delta").load(SILVER_PATH)

    print("Validation row count:", silver_df.count())
    print("Cleaned schema:")
    silver_df.printSchema()

    print("\nSample cleaned rows:")
    silver_df.show(10, truncate=False)

    if "Returned" in silver_df.columns:
        print("\nReturned target distribution:")
        silver_df.groupBy("Returned").count().orderBy("Returned").show()

finally:
    spark.stop()
