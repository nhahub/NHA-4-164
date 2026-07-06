Enhanced Pipeline for Churn Prediction (DEPI)/
├── requirements.txt
├── README.md
├── milestone1-ml-pipeline/
│   ├── Milestone_1_Data_Collection_&_Feature_Engineering.ipynb
│   ├── artifacts/
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   ├── y_test.csv
│   │   ├── minmax_scaler.pkl
│   │   └── feature_columns.json
│   └── EDA_plots/
├── milestone2-ml-pipeline/
│   ├── Milestone_2_AI_Model_Integration.ipynb
│   ├── models/
│   │   ├── xgb_churn_model.pkl
│   │   ├── rf_churn_model.pkl
│   │   └── best_churn_model.pkl
│   └── evaluation_plots/
└── milestone3-api-deployment/
    └── milestone3/
        ├── Dockerfile
        ├── docker-compose.yml
        ├── requirements.txt
        ├── train_and_save_model.py
        ├── run_pipeline.py
        ├── app/
        │   ├── main.py
        │   ├── api/
        │   ├── core/
        │   ├── models/
        │   ├── schemas/
        │   ├── services/
        │   └── data/predictions/prediction_logs.csv   ← Milestone 4 monitoring log
        └── dags/
            └── churn_inference_dag.py