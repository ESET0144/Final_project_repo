from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import json

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator


def load_data():
    db_url = "postgresql+psycopg2://postgres:postgres@host.docker.internal:5432/merged_database"
    engine = create_engine(db_url)

    query = """
        SELECT 
            meter_id,
            customer_id,
            datetime,
            energy_billed,
            daily_consumption_load
        FROM merged_table;
    """

    df = pd.read_sql(query, engine)
    df['datetime'] = pd.to_datetime(df['datetime'], format="%d-%m-%Y %H:%M")
    return df

def prepare_data(**context):
    df = load_data()
    print(f"Loaded data shape: {df.shape}")
    
    X = df[['daily_consumption_load']]
    y = df['energy_billed']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Data prepared: X_train shape {X_train.shape}, X_test shape {X_test.shape}")
    
    # Debug the actual data types
    print(f"X_train type: {type(X_train)}")
    print(f"X_train['daily_consumption_load'] type: {type(X_train['daily_consumption_load'])}")
    
    # Convert to lists explicitly
    X_train_values = X_train['daily_consumption_load'].values.tolist()
    X_test_values = X_test['daily_consumption_load'].values.tolist()
    y_train_values = y_train.values.tolist()
    y_test_values = y_test.values.tolist()
    
    print(f"X_train_values type: {type(X_train_values)}, length: {len(X_train_values)}")
    print(f"Sample X: {X_train_values[:5]}")
    print(f"Sample y: {y_train_values[:5]}")
    
    ti = context['ti']
    ti.xcom_push(key='X_train', value=str(X_train_values))  # Simple string representation
    ti.xcom_push(key='X_test', value=str(X_test_values))
    ti.xcom_push(key='y_train', value=str(y_train_values))
    ti.xcom_push(key='y_test', value=str(y_test_values))
    
    print("✅ Data successfully pushed to XCom")
    
    return "Data prepared successfully"

def train_model(**context):
    print("Starting model training...")
    
    # Pull data from XCom
    ti = context['ti']
    
    # Pull data as strings
    X_train_str = ti.xcom_pull(task_ids='prepare_data', key='X_train')
    X_test_str = ti.xcom_pull(task_ids='prepare_data', key='X_test')
    y_train_str = ti.xcom_pull(task_ids='prepare_data', key='y_train')
    y_test_str = ti.xcom_pull(task_ids='prepare_data', key='y_test')
    
    print(f"Raw X_train_str: {X_train_str}")
    print(f"Raw y_train_str: {y_train_str}")
    
    # Convert string representations back to lists using ast.literal_eval
    import ast
    X_train_list = ast.literal_eval(X_train_str)
    X_test_list = ast.literal_eval(X_test_str)
    y_train_list = ast.literal_eval(y_train_str)
    y_test_list = ast.literal_eval(y_test_str)
    
    print(f"X_train_list type: {type(X_train_list)}, length: {len(X_train_list)}")
    print(f"y_train_list type: {type(y_train_list)}, length: {len(y_train_list)}")
    
    # Convert to DataFrames/Series
    X_train = pd.DataFrame({'daily_consumption_load': X_train_list})
    X_test = pd.DataFrame({'daily_consumption_load': X_test_list})
    y_train = pd.Series(y_train_list)
    y_test = pd.Series(y_test_list)
    
    print(f"✅ Data loaded - X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Calculate metrics
    rmse = root_mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2_score_1 = r2_score(y_test, preds)

   # Save to current working directory
    import joblib
    import os
    
    model_path = './linear_regression_model.pkl'
    joblib.dump(model, model_path)
    
    # Get absolute path
    abs_path = os.path.abspath(model_path)
    print(f"✅ Model saved to: {abs_path}")
    
    # Check current directory
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    print(f"Files in current directory: {os.listdir(cwd)}")
    
    # Push metrics to XCom
    metrics = {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2_score': float(r2_score_1)
    }
    ti.xcom_push(key='metrics', value=json.dumps(metrics))
    ti.xcom_push(key='model_path', value=model_path)

    print(f"Training complete! RMSE: {rmse:.4f}, R2: {r2_score_1:.4f}")

 

    print(f"Training complete with Linear Regression.")
    print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2 Score: {r2_score_1:.4f}")
    print(f"Model saved to: {model_path}")
    print("metrics:", metrics)
    
    return "Model training completed successfully"


def validate_training(**context):
    ti = context['ti']
    metrics = json.loads(ti.xcom_pull(task_ids='train_model', key='metrics'))
    
    # Add validation logic here
    print(f"Model Metrics - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R2: {metrics['r2_score']:.4f}")
    
    # Example validation: Check if R2 score is above threshold
    if metrics['r2_score'] > 0.5:
        print("✅ Model validation passed - R2 score above threshold")
    else:
        print("⚠️ Model validation warning - R2 score below 0.5")
    
    return "Validation completed"

def log_to_mlflow(**context):
    """Log model and metrics to MLflow using the saved model"""
    ti = context['ti']
    
    print("Logging to MLflow using saved model...")
    
    try:
        # Set MLflow tracking to localhost
        mlflow.set_tracking_uri("http://localhost:8000")
        mlflow.set_experiment("meter_forecasting_exp_doc")
        
        # Pull metrics and model path from training task
        metrics_json = ti.xcom_pull(task_ids='train_model', key='metrics')
        model_path = ti.xcom_pull(task_ids='train_model', key='model_path')
        
        if metrics_json is None:
            print("❌ No metrics found in XCom")
            return "MLflow logging skipped - no metrics"
            
        if model_path is None:
            print("❌ No model path found in XCom")
            return "MLflow logging skipped - no model"
        
        metrics = json.loads(metrics_json)
        
        # Load the saved model
        import joblib
        model = joblib.load(model_path)
        
        print(f"✅ Loaded model from: {model_path}")
        
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("model_type", "LinearRegression")
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("feature", "daily_consumption_load")
            
            # Log metrics from training
            mlflow.log_metric("rmse", metrics['rmse'])
            mlflow.log_metric("mae", metrics['mae'])
            mlflow.log_metric("r2_score", metrics['r2_score'])
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log additional info
            # mlflow.log_param("training_samples", len(X_train_list))  # You might need to pull this from XCom too
            mlflow.log_param("feature_names", "daily_consumption_load")
            
            print("✅ Successfully logged to MLflow")
            print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
        
        return "MLflow logging completed successfully"
        
    except Exception as e:
        print(f"⚠️ MLflow logging failed: {e}")
        print("Continuing without MLflow...")
        return f"MLflow logging skipped: {str(e)}"

# Define the DAG
with DAG(
    'meter_forecasting_pipeline',
    description='ML Pipeline for Meter Energy Forecasting',
    tags=['ml', 'forecasting', 'energy'],
    start_date=datetime(2025, 11, 18),
    schedule="*/15 * * * *",
    catchup=False,
) as dag:

    # Task 1: Prepare data
    prepare_data_task = PythonOperator(
        task_id='prepare_data',
        python_callable=prepare_data,
    )

    # Task 2: Train model with MLflow tracking
    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )

    # Task 3: Validate training results
    validate_training_task = PythonOperator(
        task_id='validate_training',
        python_callable=validate_training,
    )

    # Task 4: Log to MLflow (uses saved model)
    mlflow_task = PythonOperator(
        task_id='log_to_mlflow',
        python_callable=log_to_mlflow,
    )

    # Simple linear dependencies
    prepare_data_task >> train_model_task >> validate_training_task >> mlflow_task