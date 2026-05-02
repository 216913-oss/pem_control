import os, joblib, requests, pandas as pd
from datetime import datetime, timedelta

def run():
    try:
        # --- CONFIGURATION ---
        channel_id = "3321400"
        ts_read_key = os.getenv("THINGSPEAK_KEY") 

        # 1. Fetch Latest Data from ThingSpeak
        ts_url = f"https://api.thingspeak.com/channels/{channel_id}/feeds/last.json?api_key={ts_read_key}"
        print(f"📡 Fetching latest feed from ThingSpeak...")
        ts_res = requests.get(ts_url)
        
        if ts_res.status_code != 200:
            print(f"❌ ThingSpeak Error {ts_res.status_code}: {ts_res.text}")
            return

        data = ts_res.json()
        v = float(data.get('field1', 0))
        i = float(data.get('field2', 0))
        t = float(data.get('field3', 0))
        print(f"✅ Sensor Data: V={v}, I={i}, T={t}")

        # 2. Model Logic
        # Ensure these filenames match your repo exactly
        model = joblib.load("ann_electrolyser_model.pkl")
        scaler = joblib.load("ann_scaler.pkl")
        
        scaled = scaler.transform([[v, i, t]])
        prediction_error = model.predict(scaled)[0]
        
        # Calculating Hydrogen production
        h2_val = float(max((i * 7.6) - prediction_error, 0))
        print(f"📈 Predicted H2: {h2_val}")

        # 3. Create Forecast Data (Overwrites every 5 mins)
        now_ts = datetime.utcnow()
        forecast_data = []
        
        # Adding current prediction
        forecast_data.append({
            "timestamp": now_ts.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "h2_value": h2_val,
            "category": "Actual"
        })

        # Generate a simple 24-hour forecast for your chart
        for hour in range(1, 25):
            future_ts = now_ts + timedelta(hours=hour)
            forecast_data.append({
                "timestamp": future_ts.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "h2_value": h2_val, # Static forecast for now
                "category": "Forecast"
            })

        # 4. Save to CSV
        df = pd.DataFrame(forecast_data)
        df.to_csv("forecast.csv", index=False)
        print("📁 forecast.csv saved successfully.")

    except Exception as e:
        print(f"💥 Critical Failure: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    run()
