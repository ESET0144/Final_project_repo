import pandas as pd
from sqlalchemy import create_engine
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

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

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:8000")
    mlflow.set_experiment("meter_forecasting_exp_doc")

    df = load_data()

    # Simple one-feature regression
    X = df[['daily_consumption_load']]
    y = df['energy_billed']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():

        # ---------------------------
        # ✔ Simple Linear Regression
        # ---------------------------
        model = LinearRegression()
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        rmse = root_mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2_score_1 = r2_score(y_test, preds)
        

        # Log params
        mlflow.log_param("model_type", "LinearRegression")

        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("R2 score", r2_score_1)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        print("Training complete with Linear Regression.")
