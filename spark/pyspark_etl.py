from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, trim, regexp_replace, to_date, when, lower, lit,
    create_map, concat_ws, sum as _sum
)
from pyspark.sql.types import StringType, DoubleType, IntegerType
import sys

def main():
    spark = (
        SparkSession.builder
        .appName("ShopStreamETL_Production")
        .enableHiveSupport()
        .getOrCreate()
    )

    try:
        raw_hdfs_path = "hdfs://hadoop-master:9000/group_project/raw/ecommerce_orders_dataset.csv"
        clean_hdfs_path = "hdfs://hadoop-master:9000/group_project/cleaned/ecommerce_orders_parquet"

        print("INFO: Starting data ingestion...")
        df = (
            spark.read.option("header", True)
            .option("escape", '"')
            .option("multiLine", False)
            .csv(raw_hdfs_path)
        )

        print("INFO: Raw data read successful")

        new_cols = [
            c.strip().replace(" ", "_").replace(".", "").replace("/", "_").replace("-", "_")
            for c in df.columns
        ]
        df = df.toDF(*new_cols)

        for f in df.schema.fields:
            if isinstance(f.dataType, StringType):
                df = df.withColumn(f.name, regexp_replace(trim(col(f.name)), r"\s+", " "))

        if all(c in df.columns for c in ["Month", "Day", "Year"]):
            df = df.withColumn(
                "Order_Date",
                to_date(
                    concat_ws(
                        "/",
                        col("Month").cast("string"),
                        col("Day").cast("string"),
                        col("Year").cast("string")
                    ),
                    "M/d/yyyy"
                )
            )

        for drop_col in ["Day", "Month", "Year"]:
            if drop_col in df.columns:
                df = df.drop(drop_col)

        if "Order_Date" in df.columns:
            bad_dates = df.select(
                _sum(when(col("Order_Date").isNull(), 1).otherwise(0)).alias("null_dates")
            ).first()["null_dates"]
            print(f"INFO: Null Order_Date rows after rebuild = {bad_dates}")

        numeric_cols_double = [
            "Unit_Price", "Discount_Amount", "Shipping_Cost", "Tax_Amount",
            "Order_Amount", "Review_Rating", "Customer_Lifetime_Value",
            "Profit_Amount", "Profit_Margin_Percent"
        ]
        numeric_cols_int = ["Quantity", "Discount_Percent", "Delivery_Days", "Quarter", "Customer_Age"]

        for c in numeric_cols_double:
            if c in df.columns:
                df = df.withColumn(c, col(c).cast(DoubleType()))

        for c in numeric_cols_int:
            if c in df.columns:
                df = df.withColumn(c, col(c).cast(IntegerType()))

        binary_cols = ["Coupon_Used", "Returned", "High_Value_Order"]
        for c in binary_cols:
            if c in df.columns:
                df = df.withColumn(
                    c,
                    when(col(c).rlike("(?i)^(yes|y|true|1)$"), 1).otherwise(0).cast(IntegerType())
                )

        lower_cols = [
            "Customer_Gender", "Payment_Method", "Device_Type",
            "Traffic_Source", "Order_Status", "Membership_Status",
            "Season", "Holiday_Season", "Shipping_Method", "Warehouse_Region",
            "Customer_Segment", "Country", "City", "Brand",
            "Product_Category", "Product_Subcategory"
        ]
        for c in lower_cols:
            if c in df.columns:
                df = df.withColumn(c, lower(col(c)))

        if "Membership_Status" in df.columns:
            membership_map = {"standard": 0, "silver": 1, "gold": 2, "platinum": 3}
            mapping_expr = create_map([lit(x) for pair in membership_map.items() for x in pair])
            df = df.withColumn("Membership_Status_ord", mapping_expr.getItem(col("Membership_Status")))

        null_counts = df.select([_sum(when(col(c).isNull(), 1).otherwise(0)).alias(c) for c in df.columns])
        null_counts.show(truncate=False)

        print("INFO: Cleaning steps completed")
        print("INFO: Writing cleaned data to HDFS...")

        df.write.mode("overwrite").parquet(clean_hdfs_path)
        df.write.mode("overwrite").option("header", True).csv("file:///home/hadoop/project/data/cleaned/")

        print("INFO: Pipeline completed successfully.")
        df.printSchema()

    except Exception as e:
        print(f"ERROR: Pipeline failed due to: {str(e)}")
        sys.exit(1)

    finally:
        spark.stop()

if __name__ == "__main__":
    main()
