import os, joblib, requests, pandas as pd, numpy as np
from datetime import datetime, timedelta

def run():
    try:
        # 1. Load Model (Look in current folder '.')
        model = joblib.load("ann_electrolyser_model.pkl")
        scaler = joblib.load("ann_scaler.pkl")
        pbi_url = os.getenv("PBI_URL")

        # 2. Get Latest State from ThingSpeak
        url = 'https://api.thingspeak.com/channels/3321400/feeds.csv?results=1'
        latest = pd.read_csv(url).iloc[-1]
        v = float(latest['field1'])
        i = float(latest['field2'])
        t = float(latest['field3'])

        # 3. Predict "Now"
        scaled_now = scaler.transform([[v, i, t]])
        err_now = model.predict(scaled_now)[0]
        h2_now = float(max((i * 7.6) - err_now, 0))

        # 4. Prepare Payload (Match Power BI exactly!)
        # Use ISO 8601 format: YYYY-MM-DDTHH:MM:SS.000Z
        now_ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")
        
        payload = [{
            "timestamp": now_ts,
            "h2_value": h2_now,
            "category": "Actual",
            "run_id": now_ts
        }]

        # 5. Push to Power BI
        print(f"DEBUG: Sending to Power BI: {payload}")
        response = requests.post(pbi_url, json=payload)

        if response.status_code == 200:
            print("🚀 Success! Data pushed to Power BI.")
        else:
            print(f"❌ Power BI Error {response.status_code}: {response.text}")
            # This will show you exactly which field name is wrong!

    except Exception as e:
        print(f"💥 Script Crashed: {e}")
        exit(1)

if __name__ == "__main__":
    run()
