import os, joblib, requests, pandas as pd, numpy as np
from datetime import datetime, timedelta

def run():
    # Load
    model = joblib.load("ann_electrolyser_model.pkl")
    scaler = joblib.load("ann_scaler.pkl")
    pbi_url = os.getenv("PBI_URL")

    # Get Latest State
    url = f'https://api.thingspeak.com/channels/3321400/feeds.csv?results=1'
    latest = pd.read_csv(url).iloc[-1]
    v, i, t = float(latest['field1']), float(latest['field2']), float(latest['field3'])

    # Predict "Now" (Actual)
    scaled_now = scaler.transform([[v, i, t]])
    err_now = model.predict(scaled_now)[0]
    h2_now = max((i * 7.6) - err_now, 0)

    # Prepare Payload
    run_id = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.000Z")
    payload = [{"timestamp": run_id, "h2_value": h2_now, "category": "Actual", "run_id": run_id}]

    # Generate 1-Hour Forecast (Replaceable data)
    for m in range(5, 65, 5):
        future_ts = (datetime.now() + timedelta(minutes=m)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        # Simulating slight temp rise for forecast
        scaled_f = scaler.transform([[v, i, t + (m*0.01)]])
        h2_f = max((i * 7.6) - model.predict(scaled_f)[0], 0)
        payload.append({"timestamp": future_ts, "h2_value": h2_f, "category": "Forecast", "run_id": run_id})

    requests.post(pbi_url, json=payload)
    print("🚀 Pushed to Power BI")

if __name__ == "__main__": run()
