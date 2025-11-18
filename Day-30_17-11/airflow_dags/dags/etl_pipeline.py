# import required libraries
from datetime import datetime
import os
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator, get_current_context

import pandas as pd
from sqlalchemy import create_engine, text



# ------------ LOCAL PC Postgres (Source) ------------
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@host.docker.internal:5432/data_ingestion"
)

# ------------ ASTRO Postgres (Target) ------------
ASTRO_DB = "postgresql+psycopg2://postgres:postgres@postgres:5432/merged_database"


# ----------------- EXTRACT -----------------

def extract_customer_data():
    """Extract only customer_data from local Postgres"""
    engine = create_engine(DB_URL)

    query = "SELECT * FROM customer_data"
    df = pd.read_sql(query, engine)

    print("Customer Data Extracted:")
    print(df.head())

    return df.to_dict(orient="records")  # XCom-safe


def extract_meter_data():
    """Extract from CSV inside the Astro container"""
    df = pd.read_csv("meter_data.csv")

    print("Meter Data Extracted:")
    print(df.head())

    return df.to_dict(orient="records")


# ----------------- TRANSFORM -----------------

def transform_data():
    ctx = get_current_context()

    customer_df = pd.DataFrame(ctx["ti"].xcom_pull("extract_customer_task"))
    meter_df = pd.DataFrame(ctx["ti"].xcom_pull("extract_meter_task"))

    # Remove energy_supplied if exists
    meter_df = meter_df.drop(columns=["energy_supplied"], errors="ignore")

    # Full outer join on meter_id
    merged = pd.merge(
        meter_df,
        customer_df,
        on="meter_id",
        how="outer"
    )

    print("Merged Data:")
    print(merged.head())

    return merged.to_dict(orient="records")


# ----------------- LOAD -----------------

def load_data():
    ctx = get_current_context()
    merged_df = pd.DataFrame(ctx["ti"].xcom_pull("transform_task"))

    engine = create_engine(ASTRO_DB)

    # Create table if not exists
    create_sql = """
    CREATE TABLE IF NOT EXISTS merged_table (
        id INT,
        meter_id VARCHAR(50),
        customer_id INT,
        name VARCHAR(100),
        location VARCHAR(100),
        mobile_no VARCHAR(20)
    );
    """

    with engine.begin() as conn:
        conn.execute(text(create_sql))

    # Append merged data
    merged_df.to_sql(
        name="merged_table",
        con=engine,
        if_exists="append",
        index=False
    )

    print("✅ Final merged data loaded into merged_database.merged_table")
    return True


# ------------------- DAG DEFINITION -------------------

with DAG(
    dag_id="final_etl_merge_dag",
    description="Extract local DB + CSV → Full Outer Join → Astro Postgres",
    start_date=datetime(2025, 11, 17),
    schedule="*/15 * * * *",
    catchup=False,
):

    extract_customer_task = PythonOperator(
        task_id="extract_customer_task",
        python_callable=extract_customer_data,
    )

    extract_meter_task = PythonOperator(
        task_id="extract_meter_task",
        python_callable=extract_meter_data,
    )

    transform_task = PythonOperator(
        task_id="transform_task",
        python_callable=transform_data,
    )

    load_task = PythonOperator(
        task_id="load_task",
        python_callable=load_data,
    )

    [extract_customer_task, extract_meter_task] >> transform_task >> load_task
