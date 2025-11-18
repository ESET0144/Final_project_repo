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
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=15),
}

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
    
    # Simple one-feature regression
    X = df[['daily_consumption_load']]
    y = df['energy_billed']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Push data to XCom for use in next tasks
    context['ti'].xcom_push(key='X_train', value=X_train.to_json())
    context['ti'].xcom_push(key='X_test', value=X_test.to_json())
    context['ti'].xcom_push(key='y_train', value=y_train.to_json())
    context['ti'].xcom_push(key='y_test', value=y_test.to_json())
    
    return "Data prepared successfully"

def train_model(**context):
    mlflow.set_tracking_uri("http://127.0.0.1:8000")
    mlflow.set_experiment("meter_forecasting_exp_doc")
    
    # Pull data from XCom
    ti = context['ti']
    X_train = pd.read_json(ti.xcom_pull(key='X_train'))
    X_test = pd.read_json(ti.xcom_pull(key='X_test'))
    y_train = pd.read_json(ti.xcom_pull(key='y_train'))
    y_test = pd.read_json(ti.xcom_pull(key='y_test'))

    with mlflow.start_run():

        model = LinearRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        rmse = root_mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2_score_1 = r2_score(y_test, preds)
        
        # Log params
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("random_state", 42)

        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2_score_1)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Push metrics to XCom for validation
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2_score_1
        }
        ti.xcom_push(key='metrics', value=json.dumps(metrics))

        print(f"Training complete with Linear Regression.")
        print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2 Score: {r2_score_1:.4f}")

def validate_training(**context):
    ti = context['ti']
    metrics = json.loads(ti.xcom_pull(key='metrics'))
    
    # Add validation logic here
    print(f"Model Metrics - RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R2: {metrics['r2_score']:.4f}")
    
    # Example validation: Check if R2 score is above threshold
    if metrics['r2_score'] > 0.5:
        print("✅ Model validation passed - R2 score above threshold")
    else:
        print("⚠️ Model validation warning - R2 score below 0.5")
    
    return "Validation completed"

# Define the DAG
with DAG(
    'meter_forecasting_pipeline',
    default_args=default_args,
    description='ML Pipeline for Meter Energy Forecasting',
    schedule_interval=timedelta(days=1),  # Run daily
    start_date=days_ago(1),
    tags=['ml', 'forecasting', 'energy'],
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

    # Define task dependencies
    prepare_data_task >> train_model_task >> validate_training_task