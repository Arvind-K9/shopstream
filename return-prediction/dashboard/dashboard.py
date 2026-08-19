# Streamlit dashboard for the Ecommerce Return Prediction project
#
# Reads reports produced by 07_compare_models_local.py:
#   /home/arvind/ecommerce_rf/reports/rf_model_comparison.csv
#   /home/arvind/ecommerce_rf/delta/rf_threshold_metrics
#
# Calls the FastAPI prediction service created by 08_rf_predict_api_local.py:
#   http://localhost:8000/predict
#
# Run:
#   source ~/spark_env/bin/activate
#   streamlit run ~/ecommerce_rf/dashboard/dashboard.py \
#     --server.address 0.0.0.0 --server.port 8501

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

PROJECT_DIR = "/home/arvind/ecommerce_rf"
REPORTS_DIR = f"{PROJECT_DIR}/reports"
DELTA_DIR = f"{PROJECT_DIR}/delta"

MODEL_COMPARISON_CSV = f"{REPORTS_DIR}/rf_model_comparison.csv"
THRESHOLD_METRICS_PATH = f"{DELTA_DIR}/rf_threshold_metrics"
API_BASE_URL = "http://localhost:8000"
AIRFLOW_URL = "http://localhost:8080"
MLFLOW_URL = "http://localhost:5000"

st.set_page_config(
    page_title="Ecommerce Return Prediction",
    page_icon="📦",
    layout="wide",
)


@st.cache_data(ttl=30)
def load_comparison() -> pd.DataFrame:
    if not os.path.isfile(MODEL_COMPARISON_CSV):
        raise FileNotFoundError(
            f"Model comparison CSV not found: {MODEL_COMPARISON_CSV}"
        )
    return pd.read_csv(MODEL_COMPARISON_CSV)


@st.cache_data(ttl=30)
def load_threshold_metrics() -> pd.DataFrame | None:
    try:
        from delta import configure_spark_with_delta_pip
        from pyspark.sql import SparkSession
    except ImportError:
        return None

    if not os.path.isdir(THRESHOLD_METRICS_PATH):
        return None

    builder = (
        SparkSession.builder
        .appName("EcommerceRF_Dashboard_Read")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    try:
        return spark.read.format("delta").load(THRESHOLD_METRICS_PATH).toPandas()
    finally:
        spark.stop()


def api_health() -> dict:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=3)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        return {"status": "unavailable", "error": str(error)}


def metric_value(dataframe: pd.DataFrame, model_keyword: str, column: str) -> float:
    row = dataframe[
        dataframe["model_name"].str.contains(model_keyword, case=False, na=False)
    ]
    if row.empty:
        return 0.0
    return float(row.iloc[0][column])


def calculate_order_values(
    unit_price: float,
    quantity: int,
    discount_percent: float,
    shipping_cost: float,
    tax_percent: float,
) -> tuple[float, float, float]:
    """Derive discount, tax, and final order values from transaction inputs."""
    gross_amount = unit_price * quantity
    discount_amount = gross_amount * (discount_percent / 100)
    taxable_amount = gross_amount - discount_amount
    tax_amount = taxable_amount * (tax_percent / 100)
    order_amount = taxable_amount + tax_amount + shipping_cost
    return round(discount_amount, 2), round(tax_amount, 2), round(order_amount, 2)


try:
    comparison_df = load_comparison()
except Exception as error:
    st.error(f"Dashboard cannot load the comparison report: {error}")
    st.stop()

health = api_health()

st.title("📦 Ecommerce Return Prediction Dashboard")
st.caption(
    "Airflow-orchestrated Spark/Delta pipeline with Random Forest model comparison and live prediction API"
)

with st.sidebar:
    st.header("Services")

    if health.get("status") == "healthy":
        st.success("Prediction API: healthy")
        st.caption(f"Model features: {health.get('feature_count', 'unknown')}")
    else:
        st.error("Prediction API: unavailable")
        st.caption(health.get("error", "Start Uvicorn on port 8000."))

    st.markdown(f"[Open Airflow]({AIRFLOW_URL})")
    st.markdown(f"[Open MLflow]({MLFLOW_URL})")

    if st.button("Refresh dashboard data"):
        st.cache_data.clear()
        st.rerun()

st.subheader("Model overview")

spark_auc = metric_value(comparison_df, "Spark", "roc_auc")
sklearn_auc = metric_value(comparison_df, "scikit", "roc_auc")
sklearn_recall = metric_value(comparison_df, "scikit", "recall")
sklearn_f1 = metric_value(comparison_df, "scikit", "f1")

metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
metric_col_1.metric("Spark ROC-AUC", f"{spark_auc:.4f}")
metric_col_2.metric("scikit-learn ROC-AUC", f"{sklearn_auc:.4f}")
metric_col_3.metric("scikit-learn recall", f"{sklearn_recall:.2%}")
metric_col_4.metric("scikit-learn F1", f"{sklearn_f1:.4f}")

st.info(
    "Recommended high-recall model: scikit-learn balanced Random Forest. "
    "Use the Spark Random Forest when high-precision review alerts are preferred."
)

left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Model metric comparison")

    chart_metrics = ["roc_auc", "accuracy", "precision", "recall", "f1"]
    available_metrics = [
        column for column in chart_metrics
        if column in comparison_df.columns
    ]

    chart_df = comparison_df[["model_name"] + available_metrics].melt(
        id_vars="model_name",
        var_name="metric",
        value_name="score",
    )

    chart_df["metric"] = chart_df["metric"].str.upper().replace(
        {
            "ROC_AUC": "ROC-AUC",
            "F1": "F1",
        }
    )

    figure = px.bar(
        chart_df,
        x="metric",
        y="score",
        color="model_name",
        barmode="group",
        text="score",
        labels={
            "metric": "Metric",
            "score": "Score",
            "model_name": "Model",
        },
        category_orders={
            "metric": ["ROC-AUC", "ACCURACY", "PRECISION", "RECALL", "F1"]
        },
        title="Spark vs scikit-learn Random Forest",
    )

    figure.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )

    figure.update_layout(
        yaxis_title="Score",
        yaxis_range=[0, 1.08],
        legend_title="Model",
        height=500,
        bargap=0.18,
        bargroupgap=0.04,
    )

    st.plotly_chart(figure, width="stretch")

with right_col:
    st.subheader("Evaluation table")
    display_columns = [
        "model_name",
        "framework",
        "class_weighting",
        "decision_threshold",
        "roc_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "recommendation",
    ]
    display_columns = [c for c in display_columns if c in comparison_df.columns]
    st.dataframe(
        comparison_df[display_columns],
        width="stretch",
        hide_index=True,
    )

st.subheader("Confusion-matrix counts")
confusion_columns = [
    "model_name",
    "true_negative",
    "false_positive",
    "false_negative",
    "true_positive",
]
if all(c in comparison_df.columns for c in confusion_columns):
    st.dataframe(
        comparison_df[confusion_columns],
        width="stretch",
        hide_index=True,
    )

st.subheader("Spark threshold tuning")
try:
    threshold_df = load_threshold_metrics()
    if threshold_df is None or threshold_df.empty:
        st.warning("Threshold metrics are not available.")
    else:
        threshold_df = threshold_df.sort_values("threshold")
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=threshold_df["threshold"],
                y=threshold_df["precision"],
                mode="lines+markers",
                name="Precision",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=threshold_df["threshold"],
                y=threshold_df["recall"],
                mode="lines+markers",
                name="Recall",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=threshold_df["threshold"],
                y=threshold_df["f1"],
                mode="lines+markers",
                name="F1",
            )
        )
        figure.update_layout(
            xaxis_title="Return-probability threshold",
            yaxis_title="Metric value",
            yaxis_range=[0, 1.05],
            legend_title="Metric",
            height=400,
        )
        st.plotly_chart(figure, width="stretch")
        st.dataframe(threshold_df, width="stretch", hide_index=True)
except Exception as error:
    st.warning(f"Could not load Spark threshold metrics: {error}")

st.subheader("Live return-risk prediction")
st.caption(
    "Calculated values update after you click Calculate order values or Predict return risk. "
    "This avoids invalid callbacks inside the Streamlit form."
)

with st.form("prediction_form"):
    form_col_1, form_col_2, form_col_3 = st.columns(3)

    with form_col_1:
        country = st.selectbox("Country", ["India", "USA", "UK", "Canada", "Other"])
        product_category = st.selectbox(
            "Product category",
            ["Electronics", "Clothing", "Home", "Beauty", "Sports"],
        )
        customer_segment = st.selectbox(
            "Customer segment", ["Regular", "New", "Premium"]
        )
        membership_status = st.selectbox(
            "Membership status", ["Standard", "Silver", "Gold", "Platinum"]
        )
        customer_age = st.number_input("Customer age", 18, 100, 31)
        quantity = st.number_input("Quantity", min_value=1, max_value=100, value=2)

    with form_col_2:
        unit_price = st.number_input("Unit price", min_value=0.0, value=499.0)
        discount_percent = st.number_input(
            "Discount percent", min_value=0.0, max_value=100.0, value=10.0
        )
        shipping_cost = st.number_input("Shipping cost", min_value=0.0, value=50.0)
        tax_percent = st.number_input(
            "Tax percent", min_value=0.0, max_value=100.0, value=10.0
        )
        delivery_days = st.number_input("Delivery days", 0, 60, 4)
        review_rating = st.number_input(
            "Review rating", min_value=0.0, max_value=5.0, value=4.0
        )

    with form_col_3:
        customer_gender = st.selectbox("Customer gender", ["Male", "Female", "Other"])
        payment_method = st.selectbox(
            "Payment method", ["Credit Card", "Debit Card", "UPI", "Cash on Delivery"]
        )
        device_type = st.selectbox("Device type", ["Mobile", "Desktop", "Tablet"])
        shipping_method = st.selectbox(
            "Shipping method", ["Standard", "Express", "Same Day"]
        )
        high_value_order = st.selectbox("High-value order", [0, 1])
        holiday_season = st.selectbox("Holiday season", ["No", "Yes"])

    calculate_clicked = st.form_submit_button("Calculate order values")
    predict_clicked = st.form_submit_button("Predict return risk", type="primary")

# Form widgets can only submit as a group. Recalculate after either submit
# button is clicked; this keeps derived amounts consistent with form values.
if calculate_clicked or predict_clicked:
    discount_amount, tax_amount, order_amount = calculate_order_values(
        float(unit_price),
        int(quantity),
        float(discount_percent),
        float(shipping_cost),
        float(tax_percent),
    )

    st.markdown("#### Calculated order values")
    calculated_col_1, calculated_col_2, calculated_col_3, calculated_col_4 = st.columns(4)
    calculated_col_1.metric("Gross amount", f"{float(unit_price) * int(quantity):,.2f}")
    calculated_col_2.metric("Discount amount", f"{discount_amount:,.2f}")
    calculated_col_3.metric("Tax amount", f"{tax_amount:,.2f}")
    calculated_col_4.metric("Final order amount", f"{order_amount:,.2f}")

if predict_clicked:
    payload = {
        "customer_gender": customer_gender,
        "country": country,
        "customer_segment": customer_segment,
        "product_category": product_category,
        "payment_method": payment_method,
        "device_type": device_type,
        "membership_status": membership_status,
        "shipping_method": shipping_method,
        "holiday_season": holiday_season,
        "customer_age": customer_age,
        "unit_price": float(unit_price),
        "quantity": int(quantity),
        "discount_percent": float(discount_percent),
        "discount_amount": discount_amount,
        "shipping_cost": float(shipping_cost),
        "tax_amount": tax_amount,
        "order_amount": order_amount,
        "delivery_days": delivery_days,
        "review_rating": review_rating,
        "high_value_order": high_value_order,
    }

    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        st.markdown("#### Prediction result")
        result_col_1, result_col_2, result_col_3 = st.columns(3)
        result_col_1.metric(
            "Return probability", f"{float(result['return_probability']):.2%}"
        )
        result_col_2.metric("Prediction", "Return" if result["prediction"] else "No return")
        result_col_3.metric("Risk label", result["risk_label"])
        st.json(result)
    except requests.RequestException as error:
        st.error(f"Prediction API request failed: {error}")
