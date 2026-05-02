import os, joblib, pandas as pd, numpy as np
from datetime import datetime, timedelta

# -----------------------------
# CONFIG
# -----------------------------
CHANNEL_ID = '3321400'
READ_KEY = os.getenv("THINGSPEAK_KEY")
CSV_FILE = "forecast.csv"

F = 96500
Vm = 24.465

# -----------------------------
# 1. GET CURRENT STATE
# -----------------------------
def get_state():
    url = f'https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.csv?api_key={READ_KEY}&results=500'
    df = pd.read_csv(url).dropna()

    df.rename(columns={
        'field1': 'V',
        'field2': 'I',
        'field3': 'T',
        'field4': 'H2_actual'
    }, inplace=True)

    df['created_at'] = pd.to_datetime(df['created_at'])
    df = df.sort_values('created_at')

    latest = df.iloc[-1]

    return {
        'V': float(latest['V']),
        'I': float(latest['I']),
        'T': float(latest['T'])
    }

# -----------------------------
# 2. FORECAST FUNCTION
# -----------------------------
def generate_monthly_forecast(state):

    model = joblib.load('ann_electrolyser_model.pkl')
    scaler = joblib.load('ann_scaler.pkl')
    features = joblib.load('feature_list.pkl')

    total_min = 30 * 24 * 60
    t = np.arange(total_min)
    is_on = (t % 90) < 60

    # -----------------------------
    # FUTURE SIMULATION
    # -----------------------------
    fut_df = pd.DataFrame({
        'V': np.where(is_on,
                      state['V'] + np.random.normal(0, 0.02, total_min),
                      1.5),

        'I': np.where(is_on,
                      state['I'] + np.random.normal(0, 0.05, total_min),
                      0.0),

        'T': state['T'] + 2 * np.sin(2 * np.pi * t / 1440)
    })

    # POWER FEATURE (required)
    fut_df['P'] = fut_df['V'] * fut_df['I']

    # -----------------------------
    # ENSURE FEATURE ORDER MATCHES TRAINING
    # -----------------------------
    input_features = fut_df[features]   # MUST be ['V','I','T','P']

    scaled = scaler.transform(input_features.values)

    # -----------------------------
    # PREDICT ERROR
    # -----------------------------
    pred_err = model.predict(scaled)

    # -----------------------------
    # PHYSICS MODEL
    # -----------------------------
    h2_theo = (fut_df['I'] / (2 * F)) * Vm * 1000 * 60

    h2_forecast = np.where(
        is_on,
        np.maximum(h2_theo - pred_err, 0),
        0
    )

    # -----------------------------
    # TIMESTAMPS
    # -----------------------------
    now = datetime.utcnow().replace(second=0, microsecond=0)

    rows = []
    for i, val in enumerate(h2_forecast):
        ts = (now + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:00")
        rows.append({
            "timestamp": ts,
            "h2_value": float(val),
            "category": "Forecast"
        })

    return pd.DataFrame(rows)

# -----------------------------
# 3. MAIN PIPELINE
# -----------------------------
if __name__ == "__main__":

    print("🔋 Fetching Electrolyser state...")
    state = get_state()

    print("🔮 Generating 1-Month Forecast...")
    new_forecast_df = generate_monthly_forecast(state)

    now_str = datetime.utcnow().replace(second=0, microsecond=0).strftime("%Y-%m-%d %H:%M:00")

    if os.path.exists(CSV_FILE):
        old_df = pd.read_csv(CSV_FILE)

        history = old_df[old_df['category'] == 'Actual']
        passed = old_df[(old_df['category'] == 'Forecast') &
                        (old_df['timestamp'] < now_str)].copy()

        passed['category'] = 'Actual'

        final_df = pd.concat([history, passed, new_forecast_df])
        final_df = final_df.drop_duplicates('timestamp', keep='last')
    else:
        final_df = new_forecast_df

    final_df = final_df.sort_values('timestamp').tail(60000)

    final_df.to_csv(CSV_FILE, index=False)

    print(f"✅ Forecast updated | Rows: {len(final_df)}")
