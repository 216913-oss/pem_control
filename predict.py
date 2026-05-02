import pandas as pd
import numpy as np
import joblib
import datetime
import matplotlib.pyplot as plt
import os

# =========================================================
# 🔥 LOAD MODEL
# =========================================================
model = joblib.load("lag_model.pkl")
scaler = joblib.load("lag_scaler.pkl")
features = joblib.load("lag_features.pkl")

# =========================================================
# 🔥 FETCH DATA
# =========================================================
import pytz

def get_recent_data(channel_id, read_key, minutes=1000):
    # Set timezone to Malaysia
    tz = pytz.timezone('Asia/Kuala_Lumpur')
    end = datetime.datetime.now(tz)
    start = end - datetime.timedelta(minutes=minutes)
    
    # ThingSpeak expects format: YYYY-MM-DD%20HH:NN:SS
    start_str = start.strftime('%Y-%m-%d%%20%H:%M:%S')
    end_str = end.strftime('%Y-%m-%d%%20%H:%M:%S')

    url = f'https://api.thingspeak.com/channels/{channel_id}/feeds.csv?api_key={read_key}&start={start_str}&end={end_str}'
    df = pd.read_csv(url)

    df.rename(columns={
        'field1': 'V',
        'field2': 'I',
        'field3': 'T',
        'field4': 'H2_actual'
    }, inplace=True)

    df['created_at'] = pd.to_datetime(df['created_at'])

    df = df.set_index('created_at')
    df = df.resample('1min').mean().fillna(0).reset_index()

    return df


# =========================================================
# 🔥 LOAD DATA
# =========================================================
df = get_recent_data('3321400', '4Q4YD3ZW21602X7L', minutes=1000)

# =========================================================
# 🔥 PHYSICS MODEL
# =========================================================
df['theo_H2'] = (df['I'] / 2 / 96500) * 24.4651 * 1000 * 60
df['target_err'] = df['theo_H2'] - df['H2_actual']

history = df.copy().reset_index(drop=True)

if len(history) < 40:
    raise ValueError("❌ Not enough data (>40 rows required)")

# =========================================================
# 🔥 LOGICAL CONTINUOUS FORECAST (NON-RECURSIVE)
# =========================================================
future_steps = 7 * 24 * 60
predictions = []

# 1. Get the last known steady-state values from your fetched data
# This ensures the forecast starts exactly where the hardware is now
latest_row = history.iloc[-1]
base_V = latest_row['V']
base_I = latest_row['I']
base_T = latest_row['T']

# 2. Extract the starting error history for the lag features
# We convert to a list for faster processing than a full DataFrame
error_history = history['target_err'].tolist()

print(f"🚀 Generating logical forecast for {future_steps} minutes...")

for step in range(future_steps):
    # -----------------------------------------------------
    # A. LOGIC: Minute-to-minute fluctuations (+/-) 
    # This prevents the "robotic" pattern seen in previous graphs
    # -----------------------------------------------------
    v_sim = base_V + np.random.uniform(-0.00489, 0.00489)
    i_sim = base_I + np.random.uniform(-0.3, 0.3)
    
    # Temperature: Diurnal cycle based on starting T
    # 1440 minutes = 24 hours
    t_sim = base_T + 1.5 * np.sin(2 * np.pi * step / 1440)
    
    # -----------------------------------------------------
    # B. LAG FEATURES (Directly from list, no dataframe shift)
    # -----------------------------------------------------
    lag1 = error_history[-1]
    lag2 = error_history[-2]
    lag5 = error_history[-5]
    lag30 = error_history[-30]
    ma10 = np.mean(error_history[-10:])
    
    # -----------------------------------------------------
    # C. ML PREDICTION
    # -----------------------------------------------------
    feat_row = np.array([[v_sim, i_sim, t_sim, lag1, lag2, lag5, lag30, ma10]])
    X_scaled = scaler.transform(feat_row)
    pred_err = model.predict(X_scaled)[0]
    
    # -----------------------------------------------------
    # D. PHYSICS RECONSTRUCTION + LOGICAL AGING
    # -----------------------------------------------------
    # We add a 0.2% efficiency drop over the week (aging_factor)
    aging_factor = 1 - (0.002 * (step / future_steps))
    theo_h2 = (i_sim / 2 / 96500) * 24.4651 * 1000 * 60
    h2_final = max(0, (theo_h2 - pred_err) * aging_factor)
    
    # -----------------------------------------------------
    # E. STORE & UPDATE
    # -----------------------------------------------------
    next_time = latest_row['created_at'] + pd.Timedelta(minutes=step + 1)
    
    predictions.append({
        'time': next_time,
        'H2_pred': h2_final
    })
    
    # Update the error history list for the next iteration's lags
    error_history.append(pred_err)

# Convert to DataFrame
pred_df = pd.DataFrame(predictions)

# =========================================================
# 🔥 SAVE WITH MERGE
# =========================================================
file_path = "H2_1week_prediction.csv"

if os.path.exists(file_path):
    old_df = pd.read_csv(file_path)
    old_df['time'] = pd.to_datetime(old_df['time'])
else:
    old_df = pd.DataFrame(columns=['time', 'H2_pred'])

pred_df['time'] = pd.to_datetime(pred_df['time'])

combined = pd.concat([old_df, pred_df])
combined = combined.sort_values('time')
combined = combined.drop_duplicates(subset='time', keep='last')

combined.to_csv(file_path, index=False)

print(f"✅ Saved predictions: {len(combined)} rows")

# =========================================================
# 🔥 PLOT
# =========================================================
def plot_forecast(pred_df):

    if pred_df.empty:
        print("❌ No data")
        return

    forecast = pred_df['H2_pred'].values

    plt.figure(figsize=(15, 6))

    plt.subplot(2, 1, 1)
    plt.plot(forecast[:1000])
    plt.title("Short-Term Forecast")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(forecast)
    plt.title("1-Week Forecast")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_forecast(pred_df)
