import joblib
import pandas as pd
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def load_data():
    db_url = "postgresql+psycopg2://postgres:postgres@host.docker.internal:5432/merged_database"
    engine = create_engine(db_url)
    query = "SELECT energy_billed, daily_consumption_load FROM merged_table;"
    df = pd.read_sql(query, engine)
    return df

if __name__ == "__main__":
    df = load_data()
    
    X = df[['daily_consumption_load']]
    y = df['energy_billed']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    joblib.dump(model, "linear_regression_model.pkl")
    print("Model trained and saved as linear_regression_model.pkl")