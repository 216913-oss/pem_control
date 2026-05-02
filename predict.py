import os, joblib, requests, pandas as pd
from datetime import datetime

def run():
    try:
        # 1. Load Files (Check current folder)
        # Ensure these names match your files in GitHub exactly
        model = joblib.load("ann_electrolyser_model.pkl")
        scaler = joblib.load("ann_scaler.pkl")
        pbi_url = os.getenv("PBI_URL")

        # 2. Fetch Data from ThingSpeak
        url = 'https://api.thingspeak.com/channels/3321400/feeds.csv?results=1'
        data = pd.read_csv(url).iloc[-1]
        v, i, t = float(data['field1']), float(data['field2']), float(data['field3'])

        # 3. Model Prediction
        scaled = scaler.transform([[v, i, t]])
        prediction_error = model.predict(scaled)[0]
        h2_val = float(max((i * 7.6) - prediction_error, 0))

        # 4. Format the "Secret" Timestamp (ISO 8601)
        # Power BI is very picky about the 'Z' at the end
        now_str = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")

        # 5. THE PAYLOAD (Matching your exact JSON structure)
        payload = [{
            "timestamp": now_str,
            "h2_value": h2_val,
            "category": "Actual", # You can change this to "Forecast" for forecast rows
            "run_id": now_str
        }]

        # 6. Send to Power BI
        print(f"DEBUG: Attempting to send: {payload}")
        response = requests.post(pbi_url, json=payload)

        if response.status_code == 200:
            print("🚀 Success! Data is now in Power BI.")
        else:
            print(f"❌ Power BI REJECTED the data (Error {response.status_code})")
            print(f"Reason: {response.text}") # THIS TELLS YOU EXACTLY WHAT IS WRONG
            exit(1)

    except Exception as e:
        print(f"💥 SCRIPT CRASHED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    run()
