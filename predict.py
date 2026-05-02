import os, joblib, requests, pandas as pd
from datetime import datetime

def run():
    try:
        # --- CONFIGURATION ---
        channel_id = "3321400"
        # Get these from GitHub Secrets
        ts_read_key = os.getenv("THINGSPEAK_KEY") 
        pbi_url = os.getenv("PBI_URL")

        # 1. Fetch Latest Data via JSON API (The "Other Way")
        ts_url = f"https://api.thingspeak.com/channels/{channel_id}/feeds/last.json?api_key={ts_read_key}"
        
        print(f"📡 Fetching latest feed from ThingSpeak...")
        ts_res = requests.get(ts_url)
        
        if ts_res.status_code != 200:
            print(f"❌ ThingSpeak Error {ts_res.status_code}: {ts_res.text}")
            return

        data = ts_res.json()
        
        # Map your sensors (Ensure field1=V, field2=I, field3=T)
        v = float(data.get('field1', 0))
        i = float(data.get('field2', 0))
        t = float(data.get('field3', 0))
        
        print(f"✅ Sensor Data: V={v}, I={i}, T={t}")

        # 2. Model Logic (Ensure these files are in your main repo folder)
        model = joblib.load("ann_electrolyser_model.pkl")
        scaler = joblib.load("ann_scaler.pkl")
        
        scaled = scaler.transform([[v, i, t]])
        prediction_error = model.predict(scaled)[0]
        h2_val = float(max((i * 7.6) - prediction_error, 0))

        # 3. Format for Power BI
        now_ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")
        
        payload = [{
            "timestamp": now_ts,
            "h2_value": h2_val,
            "category": "Actual",
            "run_id": now_ts
        }]

        # 4. Push to Power BI
        print(f"🚀 Pushing to Power BI...")
        pbi_res = requests.post(pbi_url, json=payload)
        
        if pbi_res.status_code == 200:
            print("✨ Success! Dashboard updated.")
        else:
            print(f"❌ Power BI Reject: {pbi_res.text}")

    except Exception as e:
        print(f"💥 Critical Failure: {e}")
        exit(1)

if __name__ == "__main__":
    run()
