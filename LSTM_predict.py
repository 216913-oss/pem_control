import pandas as pd
import numpy as np
import joblib
import datetime
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model

# =========================================================
# 🔥 1. LOAD LSTM MODEL & ASSETS
# =========================================================
# LSTMs are usually saved as .h5 or .keras files
model = load_model("LSTM_model.h5", compile=False)
scaler = joblib.load("LSTM_scaler.pkl")

# Note: Ensure these match what you used during training
WINDOW_SIZE = 30  # Number of past minutes the LSTM looks at
FEATURES_LIST = ['V', 'I', 'T'] 

# =========================================================
# 🔥 2. FETCH DATA (FIXED FOR MALAYSIA TIME)
# =========================================================
def get_recent_data(channel_id, read_key, minutes=1000):
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

    df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_convert('Asia/Kuala_Lumpur').dt.tz_localize(None)
    df = df.set_index('created_at')
    df = df.resample('1min').mean().fillna(method='ffill').reset_index()

    return df

ts_read_id = os.getenv('THINGSPEAK_READ_ID', '3321400')
ts_read_key = os.getenv('THINGSPEAK_READ_KEY', '4Q4YD3ZW21602X7L')
df = get_recent_data(ts_read_id, ts_read_key, minutes=1000)

# =========================================================
# 🔥 3. PHYSICS PREPARATION
# =========================================================
history = df.copy().reset_index(drop=True)
if len(history) < WINDOW_SIZE:
    raise ValueError(f"❌ Not enough data (at least {WINDOW_SIZE} rows required)")

# =========================================================
# 🔥 4. FORECAST LOOP (LSTM WINDOW LOGIC)
# =========================================================
future_steps = 7 * 24 * 60
predictions = []

latest_row = history.iloc[-1]
base_V = latest_row['V']
base_I = latest_row['I']
base_T = latest_row['T']
start_time = latest_row['created_at']

# Initialize the sliding window with the most recent actual data
# Shape: (WINDOW_SIZE, num_features)
current_window = history[FEATURES_LIST].tail(WINDOW_SIZE).values

print(f"🚀 Generating 7-day LSTM forecast...")

for step in range(future_steps):
    # A. Scale the window
    scaled_window = scaler.transform(current_window)
    
    # B. Reshape for LSTM: (1 Sample, WINDOW_SIZE steps, num_features)
    X_input = np.reshape(scaled_window, (1, WINDOW_SIZE, len(FEATURES_LIST)))
    
    # C. Predict Error (Residual)
    pred_err = model.predict(X_input, verbose=0)[0][0]
    
    # D. Physics Reconstruction
    theo = (base_I / 2 / 96500) * 24.4651 * 1000 * 60
    h2_base = theo - pred_err
    h2_final = np.clip(h2_base, 0, 65)
    
    # E. Timestamp
    next_time = start_time + pd.Timedelta(minutes=step + 1)
    predictions.append({'time': next_time, 'H2_pred': h2_final})
    
    # F. Update the sliding window for the next minute
    # We "slide" by removing the first entry and adding the current setpoints
    new_row = np.array([[base_V, base_I, base_T]])
    current_window = np.append(current_window[1:], new_row, axis=0)

# =========================================================
# 🔥 5. SMART MERGE, SAVE & PLOT
# =========================================================
pred_df = pd.DataFrame(predictions)
file_path = "LSTM_prediction.csv"

# 1. Load existing data if it exists
if os.path.exists(file_path):
    try:
        old_df = pd.read_csv(file_path)
        old_df['time'] = pd.to_datetime(old_df['time'])
        
        # 2. Combine old and new data
        # We put pred_df (new) after old_df so that new values are preferred
        combined_df = pd.concat([old_df, pred_df], ignore_index=True)
        
        # 3. Deduplicate: If timestamps overlap, keep the LAST one (the new prediction)
        # This keeps past history while updating future values with the latest forecast
        combined_df = combined_df.drop_duplicates(subset=['time'], keep='last')
        
        # 4. Sort by time and limit size (Optional: Keep last 30 days to prevent huge files)
        combined_df = combined_df.sort_values('time')
        
        # Optional: Uncomment the line below to keep only the last 45,000 minutes (~1 month)
        # combined_df = combined_df.tail(45000) 
        
        final_df = combined_df
        print(f"🔄 Merged {len(pred_df)} new predictions with existing history.")
    except Exception as e:
        print(f"⚠️ Error loading old CSV, starting fresh: {e}")
        final_df = pred_df
else:
    print("🆕 No existing CSV found. Creating new one.")
    final_df = pred_df

# 5. Save the combined results
final_df.to_csv(file_path, index=False)
print(f"✅ Saved updated rolling predictions to {file_path}")

# --- Plotting ---
def plot_forecast(df):
    plt.figure(figsize=(15, 6))
    # We plot the last 10,000 minutes so the plot doesn't get too crowded
    plot_data = df.tail(10080) 
    plt.plot(plot_data['time'], plot_data['H2_pred'], color='blue', linewidth=0.8)
    plt.title("H2 Forecast: Past History + 7-Day Future (Updated Every 5m)")
    plt.ylabel("H2 (mL/min)")
    plt.xlabel("Time")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_forecast(final_df)
