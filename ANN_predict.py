import pandas as pd
import numpy as np
import joblib
import datetime
import matplotlib.pyplot as plt
import os

# =========================================================
# 🔥 1. LOAD MODEL & ASSETS
# =========================================================
model = joblib.load("ANN_lag_model.pkl")
scaler = joblib.load("ANN_lag_scaler.pkl")
features = joblib.load("ANN_lag_features.pkl")

# =========================================================
# 🔥 2. FETCH DATA (FIXED FOR MALAYSIA TIME)
# =========================================================
def get_recent_data(channel_id, read_key, minutes=1000):
    # Force UTC+8 (Malaysia Time) for the request
    tz_offset = datetime.timezone(datetime.timedelta(hours=8))
    end = datetime.datetime.now(tz_offset)
    start = end - datetime.timedelta(minutes=minutes)

    start_str = start.strftime('%Y-%m-%d%%20%H:%M:%S')
    end_str = end.strftime('%Y-%m-%d%%20%H:%M:%S')

    url = f'https://api.thingspeak.com/channels/{channel_id}/feeds.csv?api_key={read_key}&start={start_str}&end={end_str}'
    df = pd.read_csv(url)

    df.rename(columns={
        'field1': 'V', 'field2': 'I', 'field3': 'T', 'field4': 'H2_actual'
    }, inplace=True)

    # Convert ThingSpeak UTC to Malaysia Time and strip timezone info for CSV compatibility
    df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_convert('Asia/Kuala_Lumpur').dt.tz_localize(None)

    df = df.set_index('created_at')
    df = df.resample('1min').mean().fillna(method='ffill').reset_index()

    return df

# Load Data
df = get_recent_data('3321400', '4Q4YD3ZW21602X7L', minutes=1000)

# =========================================================
# 🔥 3. PHYSICS PREPARATION
# =========================================================
df['theo_H2'] = (df['I'] / 2 / 96500) * 24.4651 * 1000 * 60
df['target_err'] = df['theo_H2'] - df['H2_actual']

history = df.copy().reset_index(drop=True)
if len(history) < 40:
    raise ValueError("❌ Not enough data (>40 rows required)")

# =========================================================
# 🔥 4. FORECAST LOOP (NO AGING)
# =========================================================
future_steps = 7 * 24 * 60
predictions = []

latest_row = history.iloc[-1]
base_V = latest_row['V']
base_I = latest_row['I']
base_T = latest_row['T']
start_time = latest_row['created_at']

# Extract starting error history for lag features
error_history = history['target_err'].tolist()

print(f"🚀 Generating forecast for {future_steps} minutes (No Aging)...")

for step in range(future_steps):
    # A. SENSOR FLUCTUATIONS (Logical Jitter)
    v_sim = base_V + np.random.uniform(-0.00489, 0.00489)
    i_sim = base_I + np.random.uniform(-0.3, 0.3)
    
    # B. DIURNAL TEMPERATURE CYCLE
    t_sim = base_T + 1.5 * np.sin(2 * np.pi * step / 1440)
    
    # C. LAG FEATURES
    lag1, lag2, lag5, lag30 = error_history[-1], error_history[-2], error_history[-5], error_history[-30]
    ma10 = np.mean(error_history[-10:])
    
    # D. ML PREDICTION
    feat_row = np.array([[v_sim, i_sim, t_sim, lag1, lag2, lag5, lag30, ma10]])
    X_scaled = scaler.transform(feat_row)
    pred_err = model.predict(X_scaled)[0]
    
    # E. PHYSICS RECONSTRUCTION (NO AGING FACTOR)
    # Applying Omron D6F-P0001A1 ±5 mL/min Tolerance logic
    theo = (i_sim / 2 / 96500) * 24.4651 * 1000 * 60
    h2_base = theo - pred_err
    h2_final = np.clip(h2_base, 0, 65)
    
    # F. TIMESTAMP (Fixed Baseline)
    next_time = start_time + pd.Timedelta(minutes=step + 1)
    
    predictions.append({'time': next_time, 'H2_pred': h2_final})
    error_history.append(pred_err)

# =========================================================
# 🔥 5. SAVE & PLOT
# =========================================================
pred_df = pd.DataFrame(predictions)
file_path = "ANN_prediction.csv"

# Force Overwrite to prevent Merge Conflicts in GitHub Actions
pred_df.to_csv(file_path, index=False)

print(f"✅ Saved predictions to {file_path}")

def plot_forecast(df):
    plt.figure(figsize=(15, 6))
    plt.plot(df['H2_pred'].values, color='blue', linewidth=0.8)
    plt.title("7-Day Continuous H2 Forecast (No Aging, Fixed Timezone)")
    plt.ylabel("H2 (mL/min)")
    plt.grid(True, alpha=0.3)
    plt.show()

plot_forecast(pred_df)
