"""
Airflow DAG: ecommerce_return_pipeline_local

Orchestrates the repeatable local batch pipeline:
  ETL -> feature engineering -> RF feature preparation -> Spark RF training
  -> threshold tuning -> scikit-learn RF training -> model comparison
  -> artifact validation

One-time infrastructure steps are deliberately excluded:
  - 01_hdfs_setup.sh
  - 02_upload_to_hdfs.sh

The FastAPI service is deliberately excluded because it is a long-running
prediction service, not a batch task. Restart it manually after retraining:
  source /home/arvind/spark_env/bin/activate
  uvicorn 08_rf_predict_api_local:app \
    --app-dir /home/arvind/ecommerce_rf/scripts \
    --host 0.0.0.0 --port 8000

Prerequisites:
  1. Airflow installed and configured.
  2. Spark environment at /home/arvind/spark_env.
  3. Project scripts stored in /home/arvind/ecommerce_rf/scripts.
  4. The source ETL script filename set correctly below.

Install example (use a separate Airflow virtual environment in practice):
  python -m pip install apache-airflow

Place this file in Airflow's DAG folder, commonly:
  ~/airflow/dags/09_ecommerce_return_pipeline_dag.py

Then start Airflow services and trigger DAG manually from the Airflow UI.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

# -----------------------------------------------------------------------
# 1. Local project configuration
# -----------------------------------------------------------------------
PROJECT_DIR = "/home/arvind/ecommerce_rf"
SCRIPTS_DIR = f"{PROJECT_DIR}/scripts"
PYTHON_BIN = "/home/arvind/spark_env/bin/python"

# Update this only if your local ETL script has a different filename.
ETL_SCRIPT = f"{SCRIPTS_DIR}/01_pyspark_etl_local.py"
FEATURE_ENGINEERING_SCRIPT = f"{SCRIPTS_DIR}/02_feature_engineering_local.py"
RF_FEATURE_PREP_SCRIPT = f"{SCRIPTS_DIR}/03_rf_feature_prep_local.py"
SPARK_RF_TRAIN_SCRIPT = f"{SCRIPTS_DIR}/04_rf_train_local.py"
THRESHOLD_TUNE_SCRIPT = f"{SCRIPTS_DIR}/05_rf_predict_tune_local.py"
SKLEARN_RF_TRAIN_SCRIPT = f"{SCRIPTS_DIR}/06_rf_train_sklearn_local.py"
MODEL_COMPARE_SCRIPT = f"{SCRIPTS_DIR}/07_compare_models_local.py"

# Durable final artifacts expected after a successful full pipeline run.
MODEL_FILE = f"{PROJECT_DIR}/models/rf_transaction_return_sklearn.pkl"
FEATURE_COLUMNS_FILE = (
    f"{PROJECT_DIR}/models/rf_transaction_return_sklearn_columns.pkl"
)
COMPARISON_REPORT = f"{PROJECT_DIR}/reports/rf_model_comparison.md"
COMPARISON_CSV = f"{PROJECT_DIR}/reports/rf_model_comparison.csv"

# -----------------------------------------------------------------------
# 2. DAG defaults
# schedule=None means manual execution for this student/local project.
# max_active_runs=1 prevents two local Spark jobs from competing for RAM.
# -----------------------------------------------------------------------
default_args = {
    "owner": "arvind",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def validate_final_artifacts() -> None:
    """Fail the DAG if final model or reporting artifacts are absent/empty."""
    import os

    required_files = [
        MODEL_FILE,
        FEATURE_COLUMNS_FILE,
        COMPARISON_REPORT,
        COMPARISON_CSV,
    ]

    missing_or_empty = [
        path
        for path in required_files
        if not os.path.isfile(path) or os.path.getsize(path) == 0
    ]

    if missing_or_empty:
        raise FileNotFoundError(
            "Pipeline completed but required final artifacts are missing or empty:\n"
            + "\n".join(missing_or_empty)
        )

    print("All final model and report artifacts exist and are non-empty:")
    for path in required_files:
        print(f"  {path} ({os.path.getsize(path)} bytes)")


with DAG(
    dag_id="ecommerce_return_pipeline_local",
    description="Local Spark/Delta ecommerce return-prediction training pipeline",
    default_args=default_args,
    start_date=datetime(2026, 8, 18),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["ecommerce", "spark", "delta", "mlflow", "random-forest", "local"],
) as dag:

    # Each task invokes the virtual-environment Python directly. This makes
    # execution independent of the shell that started Airflow.
    run_etl = BashOperator(
        task_id="run_etl_to_silver",
        bash_command=(
            f"set -euo pipefail; "
            f"test -f {ETL_SCRIPT}; "
            f"{PYTHON_BIN} {ETL_SCRIPT}"
        ),
    )

    run_feature_engineering = BashOperator(
        task_id="run_feature_engineering",
        bash_command=(
            f"set -euo pipefail; "
            f"test -f {FEATURE_ENGINEERING_SCRIPT}; "
            f"{PYTHON_BIN} {FEATURE_ENGINEERING_SCRIPT}"
        ),
    )

    run_rf_feature_prep = BashOperator(
        task_id="prepare_rf_transaction_features",
        bash_command=(
            f"set -euo pipefail; "
            f"test -f {RF_FEATURE_PREP_SCRIPT}; "
            f"{PYTHON_BIN} {RF_FEATURE_PREP_SCRIPT}"
        ),
    )

    run_spark_rf_training = BashOperator(
        task_id="train_spark_random_forest",
        bash_command=(
            f"set -euo pipefail; "
            f"test -f {SPARK_RF_TRAIN_SCRIPT}; "
            f"{PYTHON_BIN} {SPARK_RF_TRAIN_SCRIPT}"
        ),
    )

    run_threshold_tuning = BashOperator(
        task_id="tune_spark_rf_threshold",
        bash_command=(
            f"set -euo pipefail; "
            f"test -f {THRESHOLD_TUNE_SCRIPT}; "
            f"{PYTHON_BIN} {THRESHOLD_TUNE_SCRIPT}"
        ),
    )

    run_sklearn_rf_training = BashOperator(
        task_id="train_sklearn_balanced_random_forest",
        bash_command=(
            f"set -euo pipefail; "
            f"test -f {SKLEARN_RF_TRAIN_SCRIPT}; "
            f"{PYTHON_BIN} {SKLEARN_RF_TRAIN_SCRIPT}"
        ),
    )

    run_model_comparison = BashOperator(
        task_id="compare_models_and_write_report",
        bash_command=(
            f"set -euo pipefail; "
            f"test -f {MODEL_COMPARE_SCRIPT}; "
            f"{PYTHON_BIN} {MODEL_COMPARE_SCRIPT}"
        ),
    )

    validate_artifacts = PythonOperator(
        task_id="validate_final_model_and_reports",
        python_callable=validate_final_artifacts,
    )

    (
        run_etl
        >> run_feature_engineering
        >> run_rf_feature_prep
        >> run_spark_rf_training
        >> run_threshold_tuning
        >> run_sklearn_rf_training
        >> run_model_comparison
        >> validate_artifacts
    )
