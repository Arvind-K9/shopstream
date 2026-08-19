# Local WSL PySpark + Delta Lake feature engineering (path-based version)
#
# Reads Delta layer:
#   /home/arvind/ecommerce_rf/delta/silver_orders
#
# Writes Delta layers:
#   /home/arvind/ecommerce_rf/delta/customer_features
#   /home/arvind/ecommerce_rf/delta/product_features
#   /home/arvind/ecommerce_rf/delta/customer_ml_features
#
# Run:
#   source ~/spark_env/bin/activate
#   python ~/ecommerce_rf/scripts/02_feature_engineering_local.py

import os

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# -----------------------------------------------------------------------
# 1. Project paths
# -----------------------------------------------------------------------
PROJECT_DIR = "/home/arvind/ecommerce_rf"
WAREHOUSE_DIR = f"{PROJECT_DIR}/warehouse"
DELTA_DIR = f"{PROJECT_DIR}/delta"

SILVER_PATH = f"{DELTA_DIR}/silver_orders"
CUSTOMER_FEATURES_PATH = f"{DELTA_DIR}/customer_features"
PRODUCT_FEATURES_PATH = f"{DELTA_DIR}/product_features"
CUSTOMER_ML_FEATURES_PATH = f"{DELTA_DIR}/customer_ml_features"

# -----------------------------------------------------------------------
# 2. Validate input and ensure output directory exists
# -----------------------------------------------------------------------
if not os.path.isdir(SILVER_PATH):
    raise FileNotFoundError(
        f"Silver Delta layer not found: {SILVER_PATH}\n"
        "Run 01_etl_local.py successfully before feature engineering."
    )

os.makedirs(WAREHOUSE_DIR, exist_ok=True)
os.makedirs(DELTA_DIR, exist_ok=True)

# -----------------------------------------------------------------------
# 3. Start local Spark with Delta Lake enabled
# -----------------------------------------------------------------------
builder = (
    SparkSession.builder
    .appName("EcommerceRF_FeatureEngineering_Local")
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
    # 4. Read cleaned Silver data from persistent Delta storage
    # -------------------------------------------------------------------
    df = spark.read.format("delta").load(SILVER_PATH)

    print("Silver input path:", SILVER_PATH)
    print("Input row count:", df.count())

    required_cols = [
        "Customer_ID", "Order_ID", "Order_Amount", "Order_Date", "Product_ID",
        "Product_Category", "Product_Subcategory", "Brand", "Quantity", "Returned",
        "Review_Rating", "Profit_Margin_Percent", "Customer_Segment",
        "Customer_Lifetime_Value", "Membership_Status", "Customer_Age",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # -------------------------------------------------------------------
    # 5. Customer-level aggregations
    # -------------------------------------------------------------------
    customer_base = (
        df.groupBy("Customer_ID")
        .agg(
            F.count("Order_ID").alias("total_orders"),
            F.sum("Order_Amount").alias("total_spend"),
            F.avg("Order_Amount").alias("avg_order_value"),
            F.max("Order_Date").alias("last_order_date"),
            F.min("Order_Date").alias("first_order_date"),
            F.countDistinct("Product_Category").alias("distinct_categories"),
            F.countDistinct("Product_ID").alias("distinct_products"),
            F.sum("Returned").alias("total_returns"),
            F.avg("Review_Rating").alias("avg_review_rating"),
            F.sum("Quantity").alias("total_quantity_purchased"),
            F.first("Customer_Segment", ignorenulls=True).alias("customer_segment"),
            F.first("Customer_Lifetime_Value", ignorenulls=True).alias(
                "customer_lifetime_value"
            ),
            F.first("Membership_Status", ignorenulls=True).alias("membership_status"),
            F.first("Customer_Age", ignorenulls=True).alias("customer_age_raw"),
        )
    )

    max_date = df.agg(F.max("Order_Date").alias("max_date")).first()["max_date"]

    customer_features = (
        customer_base
        .withColumn("customer_age", F.col("customer_age_raw").cast("int"))
        .withColumn(
            "age_bucket",
            F.when(F.col("customer_age") < 18, "Under 18")
            .when(F.col("customer_age").between(18, 24), "18-24")
            .when(F.col("customer_age").between(25, 34), "25-34")
            .when(F.col("customer_age").between(35, 44), "35-44")
            .when(F.col("customer_age").between(45, 54), "45-54")
            .when(F.col("customer_age").between(55, 64), "55-64")
            .when(F.col("customer_age") >= 65, "65+")
            .otherwise("Unknown"),
        )
        .withColumn(
            "customer_tenure_days",
            F.datediff(F.col("last_order_date"), F.col("first_order_date")),
        )
        .withColumn(
            "recency_days",
            F.datediff(F.lit(max_date), F.col("last_order_date")),
        )
        .drop("customer_age_raw")
    )

    # -------------------------------------------------------------------
    # 6. Global RFM scores
    # Global windows rank each customer against all customers. Spark may
    # display a no-partition warning; expected for global RFM ranking.
    # -------------------------------------------------------------------
    r_window = Window.orderBy(F.col("recency_days").asc())
    f_window = Window.orderBy(F.col("total_orders").asc())
    m_window = Window.orderBy(F.col("total_spend").asc())

    customer_features = (
        customer_features
        .withColumn("r_score", F.lit(6) - F.ntile(5).over(r_window))
        .withColumn("f_score", F.ntile(5).over(f_window))
        .withColumn("m_score", F.ntile(5).over(m_window))
        .withColumn(
            "rfm_score",
            F.col("r_score") + F.col("f_score") + F.col("m_score"),
        )
        .withColumn(
            "rfm_segment",
            F.when(F.col("rfm_score") >= 13, "Champions")
            .when(F.col("rfm_score") >= 10, "Loyal Customers")
            .when(F.col("rfm_score") >= 7, "Potential Loyalists")
            .when(F.col("rfm_score") >= 4, "At Risk")
            .otherwise("Lost / Churned"),
        )
    )

    # -------------------------------------------------------------------
    # 7. Product-level features
    # -------------------------------------------------------------------
    product_features = (
        df.groupBy("Product_ID", "Product_Category", "Product_Subcategory", "Brand")
        .agg(
            F.count("Order_ID").alias("times_ordered"),
            F.sum("Quantity").alias("total_quantity_sold"),
            F.sum("Order_Amount").alias("total_revenue"),
            F.avg("Review_Rating").alias("avg_review_rating"),
            F.sum("Returned").alias("total_returns"),
            F.avg("Profit_Margin_Percent").alias("avg_profit_margin"),
        )
    )

    # -------------------------------------------------------------------
    # 8. Customer ML / purchase-behavior features
    # -------------------------------------------------------------------
    customer_product_agg = (
        df.groupBy("Customer_ID")
        .agg(
            F.countDistinct("Product_ID").alias("ml_distinct_products_bought"),
            F.countDistinct("Brand").alias("ml_distinct_brands_bought"),
            F.sum("Quantity").alias("ml_total_quantity_bought"),
            F.avg("Quantity").alias("ml_avg_quantity_per_line"),
            F.avg("Profit_Margin_Percent").alias("ml_avg_profit_margin_purchased"),
            F.avg("Review_Rating").alias("ml_avg_product_rating_purchased"),
            F.sum(F.when(F.col("Returned") == 1, 1).otherwise(0)).alias(
                "ml_returned_line_count"
            ),
        )
    )

    top_category = (
        df.groupBy("Customer_ID", "Product_Category")
        .agg(F.count("Order_ID").alias("category_order_count"))
    )

    top_category_window = (
        Window.partitionBy("Customer_ID")
        .orderBy(F.desc("category_order_count"), F.asc("Product_Category"))
    )

    customer_top_category = (
        top_category
        .withColumn("rn", F.row_number().over(top_category_window))
        .filter(F.col("rn") == 1)
        .select(
            "Customer_ID",
            F.col("Product_Category").alias("favorite_product_category"),
            F.col("category_order_count").alias("favorite_category_order_count"),
        )
    )

    customer_ml_features = (
        customer_features
        .join(customer_product_agg, on="Customer_ID", how="left")
        .join(customer_top_category, on="Customer_ID", how="left")
    )

    # -------------------------------------------------------------------
    # 9. Write feature layers to persistent Delta paths
    # -------------------------------------------------------------------
    (
        customer_features.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(CUSTOMER_FEATURES_PATH)
    )

    (
        product_features.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(PRODUCT_FEATURES_PATH)
    )

    (
        customer_ml_features.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(CUSTOMER_ML_FEATURES_PATH)
    )

    # -------------------------------------------------------------------
    # 10. Validate by reading every output back from disk
    # -------------------------------------------------------------------
    customer_features_check = spark.read.format("delta").load(CUSTOMER_FEATURES_PATH)
    product_features_check = spark.read.format("delta").load(PRODUCT_FEATURES_PATH)
    customer_ml_features_check = (
        spark.read.format("delta").load(CUSTOMER_ML_FEATURES_PATH)
    )

    print("\n===== FEATURE ENGINEERING COMPLETE =====")
    print("Customer features path:", CUSTOMER_FEATURES_PATH)
    print("Customer features rows:", customer_features_check.count())
    print("Product features path:", PRODUCT_FEATURES_PATH)
    print("Product features rows:", product_features_check.count())
    print("Customer ML features path:", CUSTOMER_ML_FEATURES_PATH)
    print("Customer ML features rows:", customer_ml_features_check.count())

    print("\nCustomer feature sample:")
    customer_features_check.show(10, truncate=False)

    print("\nRFM segment distribution:")
    (
        customer_features_check
        .groupBy("rfm_segment")
        .count()
        .orderBy(F.desc("count"))
        .show(truncate=False)
    )

finally:
    spark.stop()
