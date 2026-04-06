conda activate mlops-local-demo
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
