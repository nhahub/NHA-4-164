# Enhanced Pipeline for Churn Prediction

## Project Overview
This project is an end-to-end MLOps pipeline for predicting customer churn using XGBoost.

The system includes:
- FastAPI REST API
- Dockerized deployment
- Apache Airflow orchestration
- Prediction logging
- Monitoring dashboard using Power BI

---

## Technologies Used
- Python
- FastAPI
- XGBoost
- Docker
- Docker Compose
- Apache Airflow
- Power BI
- Pandas

---

## Project Structure

app/
dags/
docker-compose.yml
requirements.txt
README.md

---

## How to Run the Project

### 1. Build & Run Docker Containers

docker compose up --build

---

## API Access

### Swagger Docs
http://localhost:8000/docs




### Airflow Dashboard
http://localhost:8082



Username:



Password:



---

## Prediction Endpoint

### POST `/predict`

Example Request:

```json
{
  "tenure": 24,
  "MonthlyCharges": 120,
  "TotalCharges": 3000,
  "Contract": 0,
  "total_services": 6
}
```

---

## Prediction Logging

All predictions are automatically stored in:

data/predictions/prediction_logs.csv




The logs include:
- Timestamp
- Prediction result
- Churn probability
- Customer features

---

## Monitoring Dashboard

Power BI dashboard was created using prediction_logs.csv to monitor:
- Churn distribution
- Churn probability over time
- Prediction statistics

---

## Team Members
- Ali (Team Leader)
- Youseif
- Mariam
- Mariam
- Jana
-Mohamed

---

## Future Improvements
- Model retraining pipeline
- Real-time monitoring
- Grafana integration
- CI/CD pipeline