import os, joblib, requests, pandas as pd
from datetime import datetime

def run():
    try:
        # --- NEW ROBUST DATA FETCHING ---
        # Get these from your GitHub Secrets
        channel_id = "3321400"
        api_key = os.getenv("THINGSPEAK_KEY") # Add this to GitHub Secrets!
        pbi_url = os.getenv("PBI_URL")

        # Use the JSON API (much more stable than CSV)
        ts_url = f"https://api.thingspeak.com/channels/{channel_id}/feeds/last.json?api_key={api_key}"
        
        print(f"📡 Fetching data from ThingSpeak...")
        ts_res = requests.get(ts_url)
        
        if ts_res.status_code != 200:
            print(f"❌ ThingSpeak Error {ts_res.status_code}: {ts_res.text}")
            return # Stop here if we can't get data

        data = ts_res.json()
        
        # Map fields (Adjust field numbers to match your ThingSpeak setup)
        v = float(data.get('field1', 0))
        i = float(data.get('field2', 0))
        t = float(data.get('field3', 0))
        # --------------------------------
        
        print(f"✅ Data received: V={v}, I={i}, T={t}")


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
