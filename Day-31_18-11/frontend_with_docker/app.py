from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd

# Load model
model = joblib.load('linear_regression_model.pkl')

app = FastAPI()

@app.get("/")
async def home():
    return HTMLResponse("""
    <html>
    <body>
        <h2>Energy Forecaster</h2>
        <form method="post" action="/predict">
            <input type="number" name="consumption" step="0.01" placeholder="Enter consumption" required>
            <button type="submit">Predict Bill</button>
        </form>
    </body>
    </html>
    """)

@app.post("/predict")
async def predict(consumption: float = Form(...)):
    input_data = pd.DataFrame({'daily_consumption_load': [consumption]})
    prediction = model.predict(input_data)[0]
    
    return HTMLResponse(f"""
    <html>
    <body>
        <h2>Prediction Result</h2>
        <p>Input: {consumption} kWh</p>
        <p><strong>Predicted Bill: ${round(prediction, 2)}</strong></p>
        <a href="/">Back</a>
    </body>
    </html>
    """)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)