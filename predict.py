import os, joblib, requests, pandas as pd, numpy as np
from datetime import datetime, timedelta

# --- CONFIGURATION ---
CHANNEL_ID = '3321400'
READ_KEY = os.getenv("THINGSPEAK_KEY") 
CSV_FILE = "forecast.csv"

def get_state():
    url = f'https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.csv?api_key={READ_KEY}&results=500'
    df = pd.read_csv(url).dropna()
    df.rename(columns={'field1': 'V', 'field2': 'I', 'field3': 'T'}, inplace=True)
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    df = df.sort_values('created_at').reset_index(drop=True)
    is_running = df['I'] > 0.5
    cycle_pos = 0
    for i in range(1, len(df)):
        if is_running.iloc[i]:
            dt = (df.loc[i, 'created_at'] - df.loc[i-1, 'created_at']).total_seconds() / 60
            cycle_pos += dt
        else: cycle_pos = 0
            
    latest = df.iloc[-1]
    return {'V': float(latest['V']), 'I': float(latest['I']), 'T': float(latest['T']), 'cycle_pos': cycle_pos}

def generate_monthly_forecast(state):
    model = joblib.load('ann_electrolyser_model.pkl')
    scaler = joblib.load('ann_scaler.pkl')
    
    # 1 month = 30 days * 24 hours * 60 minutes = 43,200 rows
    total_min = 30 * 24 * 60
    t = np.arange(total_min)
    
    # Electrolyser Duty Cycle (e.g., 60 mins ON, 30 mins OFF)
    is_on = (t % 90) < 60
    
    # Simulate future conditions based on current state
    fut_df = pd.DataFrame({
        'V': np.where(is_on, state['V'] + np.random.normal(0, 0.02, total_min), 1.5),
        'I': np.where(is_on, state['I'] + np.random.normal(0, 0.05, total_min), 0.0),
        'T': state['T'] + 2*np.sin(2*np.pi*t/1440), # Diurnal temp swing
        'cycle_pos': state['cycle_pos'] + t,
        'power': 0.0
    })
    fut_df['power'] = fut_df['V'] * fut_df['I']
    
    # ANN Prediction
    scaled = scaler.transform(fut_df.values)
    pred_err = ann_model.predict(scaled)
    h2_forecast = np.where(is_on, np.maximum((fut_df['I'] * 7.6) - pred_err, 0), 0)
    
    # Generate rows with seconds set to :00
    now = datetime.utcnow().replace(second=0, microsecond=0)
    new_rows = []
    for i, val in enumerate(h2_forecast):
        # We save EVERY minute now
        ts = (now + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:00")
        new_rows.append({"timestamp": ts, "h2_value": float(val), "category": "Forecast"})
    
    return pd.DataFrame(new_rows)

if __name__ == "__main__":
    print("🔋 Fetching Electrolyser state...")
    state = get_state()
    
    print("🔮 Generating 1-Month Forecast (43,200 data points)...")
    new_forecast_df = generate_monthly_forecast(state)
    
    now_str = datetime.utcnow().replace(second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:00")

    if os.path.exists(CSV_FILE):
        old_df = pd.read_csv(CSV_FILE)
        # Keep Actuals and convert passed forecasts to Actuals
        history = old_df[old_df['category'] == 'Actual']
        passed_forecasts = old_df[(old_df['category'] == 'Forecast') & (old_df['timestamp'] < now_str)].copy()
        passed_forecasts['category'] = 'Actual'
        
        # Merge and remove duplicates (keep newest AI prediction)
        final_df = pd.concat([history, passed_forecasts, new_forecast_df]).drop_duplicates('timestamp', keep='last')
    else:
        final_df = new_forecast_df

    # Safety Cap: 60,000 rows (Approx 1 month forecast + 11 days of history)
    # This prevents the CSV from growing until it crashes your Power BI Desktop
    final_df = final_df.sort_values('timestamp').tail(60000) 
    
    final_df.to_csv(CSV_FILE, index=False)
    print(f"✅ Success! Master CSV updated. Total rows: {len(final_df)}")
