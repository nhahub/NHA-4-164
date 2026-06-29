import os
import sys
import logging

# Ensure root workspace is in the python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("local_pipeline_runner")

try:
    # Import tasks directly from DAG definition file
    from dags.churn_inference_dag import (
        mock_raw_data_task,
        feature_engineering_task,
        batch_inference_task
    )
except ImportError as e:
    logger.error(f"Failed to import pipeline tasks from dags.churn_inference_dag: {str(e)}")
    sys.exit(1)

def main():
    logger.info("==============================================================")
    logger.info("   STARTING LOCAL MLOPS BATCH PREDICTION PIPELINE RUNNER      ")
    logger.info("==============================================================")
    
    try:
        # Override paths to map locally outside the docker container
        os.makedirs(os.path.join(CURRENT_DIR, "data", "inputs"), exist_ok=True)
        os.makedirs(os.path.join(CURRENT_DIR, "data", "predictions"), exist_ok=True)
        
        # Override global paths in dags.churn_inference_dag module to match host folder
        import dags.churn_inference_dag as dag_mod
        dag_mod.RAW_DATA_PATH = os.path.join(CURRENT_DIR, "data", "inputs", "raw_daily_data.csv")
        dag_mod.ENGINEERED_DATA_PATH = os.path.join(CURRENT_DIR, "data", "inputs", "engineered_daily_data.csv")
        dag_mod.RESULTS_PATH = os.path.join(CURRENT_DIR, "data", "predictions", "daily_results.csv")
        dag_mod.PRIMARY_MODEL_PATH = os.path.join(CURRENT_DIR, "app", "models", "xgb_churn_model.pkl")

        logger.info("Executing Task 1: Staging Mock Customer Data...")
        mock_raw_data_task()

        logger.info("Executing Task 2: Performing Feature Engineering...")
        feature_engineering_task()

        logger.info("Executing Task 3: Running XGBoost Batch Predictive Inference...")
        batch_inference_task()

        logger.info("==============================================================")
        logger.info("   PIPELINE SUCCESS: Batch Predictions Saved Successfully!    ")
        logger.info(f"   Destination: {dag_mod.RESULTS_PATH} ")
        logger.info("==============================================================")

    except Exception as e:
        logger.error(f"Pipeline execution encountered a fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
