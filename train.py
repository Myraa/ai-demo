# train.py
import mlflow
import tensorflow as tf

def train_model():
    # Load and preprocess data, define model, etc.
    # ...

    # Start MLflow run
    with mlflow.start_run():  
        # Log parameters, metrics, model, etc.
        # ...
        pass


if __name__ == "__main__":
    train_model()
