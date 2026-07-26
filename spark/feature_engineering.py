from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
import sys


def main():
    spark = (
        SparkSession.builder
        .appName("ShopStreamFeatureEngineering")
        .enableHiveSupport()
        .getOrCreate()
    )

    try:
        clean_hdfs_path = "hdfs://hadoop-master:9000/group_project/cleaned/ecommerce_orders_parquet"
        feature_hdfs_path = "hdfs://hadoop-master:9000/group_project/features"
        local_feature_path = "file:///home/hadoop/project/data/features"

        print("INFO: Reading cleaned data from HDFS...")
        df = spark.read.parquet(clean_hdfs_path)

        required_cols = [
            "Customer_ID", "Order_ID", "Order_Amount", "Order_Date", "Product_ID",
            "Product_Category", "Product_Subcategory", "Brand", "Quantity", "Returned",
            "Review_Rating", "Profit_Margin_Percent", "Customer_Segment",
            "Customer_Lifetime_Value", "Membership_Status", "Customer_Age"
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise Exception(f"Missing required columns: {', '.join(missing)}")

        print("INFO: Building customer-level features...")
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
                F.first("Customer_Lifetime_Value", ignorenulls=True).alias("customer_lifetime_value"),
                F.first("Membership_Status", ignorenulls=True).alias("membership_status"),
                F.first("Customer_Age", ignorenulls=True).alias("customer_age_raw")
            )
        )

        max_date = df.agg(F.max("Order_Date").alias("max_date")).first()["max_date"]

        customer_features = (
            customer_base
            .withColumn("customer_age", F.col("customer_age_raw").cast("int"))
            .withColumn(
                "age_bucket",
                F.when(F.col("customer_age") < 18, "Under 18")
                .when((F.col("customer_age") >= 18) & (F.col("customer_age") <= 24), "18-24")
                .when((F.col("customer_age") >= 25) & (F.col("customer_age") <= 34), "25-34")
                .when((F.col("customer_age") >= 35) & (F.col("customer_age") <= 44), "35-44")
                .when((F.col("customer_age") >= 45) & (F.col("customer_age") <= 54), "45-54")
                .when((F.col("customer_age") >= 55) & (F.col("customer_age") <= 64), "55-64")
                .when(F.col("customer_age") >= 65, "65+")
                .otherwise("Unknown")
            )
            .withColumn("customer_tenure_days", F.datediff(F.col("last_order_date"), F.col("first_order_date")))
            .withColumn("recency_days", F.datediff(F.lit(max_date), F.col("last_order_date")))
            .drop("customer_age_raw")
        )

        r_window = Window.orderBy(F.col("recency_days").asc())
        f_window = Window.orderBy(F.col("total_orders").asc())
        m_window = Window.orderBy(F.col("total_spend").asc())

        customer_features = (
            customer_features
            .withColumn("r_score", F.ntile(5).over(r_window))
            .withColumn("f_score", F.ntile(5).over(f_window))
            .withColumn("m_score", F.ntile(5).over(m_window))
            .withColumn("r_score", F.lit(6) - F.col("r_score"))
            .withColumn("rfm_score", F.col("r_score") + F.col("f_score") + F.col("m_score"))
            .withColumn(
                "rfm_segment",
                F.when(F.col("rfm_score") >= 13, "Champions")
                .when((F.col("rfm_score") >= 10) & (F.col("rfm_score") < 13), "Loyal Customers")
                .when((F.col("rfm_score") >= 7) & (F.col("rfm_score") < 10), "Potential Loyalists")
                .when((F.col("rfm_score") >= 4) & (F.col("rfm_score") < 7), "At Risk")
                .otherwise("Lost / Churned")
            )
        )

        print("INFO: Building product-level features...")
        product_features = (
            df.groupBy("Product_ID", "Product_Category", "Product_Subcategory", "Brand")
            .agg(
                F.count("Order_ID").alias("times_ordered"),
                F.sum("Quantity").alias("total_quantity_sold"),
                F.sum("Order_Amount").alias("total_revenue"),
                F.avg("Review_Rating").alias("avg_review_rating"),
                F.sum("Returned").alias("total_returns"),
                F.avg("Profit_Margin_Percent").alias("avg_profit_margin")
            )
        )

        print("INFO: Building customer ML master table...")
        customer_product_agg = (
            df.groupBy("Customer_ID")
            .agg(
                F.countDistinct("Product_ID").alias("ml_distinct_products_bought"),
                F.countDistinct("Brand").alias("ml_distinct_brands_bought"),
                F.sum("Quantity").alias("ml_total_quantity_bought"),
                F.avg("Quantity").alias("ml_avg_quantity_per_line"),
                F.avg("Profit_Margin_Percent").alias("ml_avg_profit_margin_purchased"),
                F.avg("Review_Rating").alias("ml_avg_product_rating_purchased"),
                F.sum(F.when(F.col("Returned") == 1, 1).otherwise(0)).alias("ml_returned_line_count")
            )
        )

        top_category = (
            df.groupBy("Customer_ID", "Product_Category")
            .agg(F.count("Order_ID").alias("category_order_count"))
        )
        top_category_window = Window.partitionBy("Customer_ID").orderBy(F.desc("category_order_count"), F.asc("Product_Category"))
        customer_top_category = (
            top_category
            .withColumn("rn", F.row_number().over(top_category_window))
            .filter(F.col("rn") == 1)
            .select(
                "Customer_ID",
                F.col("Product_Category").alias("favorite_product_category"),
                F.col("category_order_count").alias("favorite_category_order_count")
            )
        )

        customer_ml_features = (
            customer_features
            .join(customer_product_agg, on="Customer_ID", how="left")
            .join(customer_top_category, on="Customer_ID", how="left")
        )

        print("INFO: Writing feature datasets to HDFS...")
        customer_features.write.mode("overwrite").parquet(f"{feature_hdfs_path}/customer_features")
        product_features.write.mode("overwrite").parquet(f"{feature_hdfs_path}/product_features")
        customer_ml_features.write.mode("overwrite").parquet(f"{feature_hdfs_path}/customer_ml_features")

        print("INFO: Writing local CSV extracts...")
        customer_features.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{local_feature_path}/customer_features")
        product_features.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{local_feature_path}/product_features")
        customer_ml_features.coalesce(1).write.mode("overwrite").option("header", True).csv(f"{local_feature_path}/customer_ml_features")

        print("INFO: Saving feature tables to Hive...")
        spark.sql("CREATE DATABASE IF NOT EXISTS group_project")
        spark.sql("USE group_project")

        customer_features.write.mode("overwrite").saveAsTable("group_project.customer_features")
        product_features.write.mode("overwrite").saveAsTable("group_project.product_features")
        customer_ml_features.write.mode("overwrite").saveAsTable("group_project.customer_ml_features")

        print("INFO: Feature engineering complete.")
        print("Customer features rows:", customer_features.count())
        print("Product features rows:", product_features.count())
        print("Customer ML features rows:", customer_ml_features.count())

    except Exception as e:
        print(f"ERROR: Feature pipeline failed due to: {str(e)}")
        sys.exit(1)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
