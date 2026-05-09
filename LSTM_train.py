import pandas as pd
import numpy as np
import joblib
import requests
import datetime
import os

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

# TensorFlow / Keras for Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =========================================================
# 1. FETCH 1-MONTH DATA (Same as your RF script)
# =========================================================
def get_one_month_data(channel_id, read_key):
    all_chunks = []
    end_date = datetime.datetime.now()
    for i in range(14):
        chunk_end = end_date - datetime.timedelta(days=i)
        chunk_start = end_date - datetime.timedelta(days=i+1)
        start_str = chunk_start.strftime('%Y-%m-%d%%20%H:%M:%S')
        end_str = chunk_end.strftime('%Y-%m-%d%%20%H:%M:%S')
        url = f'https://api.thingspeak.com/channels/{channel_id}/feeds.csv?api_key={read_key}&start={start_str}&end={end_str}'
        try:
            chunk_df = pd.read_csv(url)
            if not chunk_df.empty: all_chunks.append(chunk_df)
        except: pass
    return pd.concat(all_chunks).drop_duplicates().reset_index(drop=True)

# =========================================================
# 2. LOAD & CLEAN DATA
# =========================================================
ts_read_id = os.getenv('THINGSPEAK_READ_ID', '3321400')
ts_read_key = os.getenv('THINGSPEAK_READ_KEY', '4Q4YD3ZW21602X7L')
ts_write_key = os.getenv('THINGSPEAK_WRITE_KEY', 'F6NHRHZ60PHVSXBP')

df = get_one_month_data(ts_read_id, ts_read_key)
df.rename(columns={'field1':'V', 'field2':'I', 'field3':'T', 'field4':'H2_actual'}, inplace=True)
df['created_at'] = pd.to_datetime(df['created_at'])
df = df.sort_values('created_at').set_index('created_at')
df = df.resample('1min').mean().dropna().reset_index()

# =========================================================
# 3. PHYSICS + TARGET ERROR
# =========================================================
df['theo_H2'] = (df['I'] / 2 / 96500) * 24.4651 * 1000 * 60
df['target_err'] = df['theo_H2'] - df['H2_actual']

# =========================================================
# 4. SEQUENCE GENERATION (CRITICAL FOR LSTM)
# =========================================================
# Instead of manual lags, we create a 3D window
def create_sequences(data, target, window_size=10):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        y.append(target[i + window_size])
    return np.array(X), np.array(y)

WINDOW_SIZE = 10 # Model looks at the last 10 minutes to predict the next
features = ['V', 'I', 'T'] # Added target_err to features so it learns from history

# Scale data before creating sequences
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features])

X, y = create_sequences(df_scaled, df['target_err'].values, WINDOW_SIZE)

# =========================================================
# 5. TRAIN / TEST SPLIT
# =========================================================
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =========================================================
# 6. LSTM MODEL ARCHITECTURE
# =========================================================
model = Sequential([
    LSTM(64, activation='relu', input_shape=(WINDOW_SIZE, len(features)), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1) # Predicts the target_err for t+1
])

model.compile(optimizer='adam', loss='mse')

print("🧠 Training LSTM MODEL...")
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# =========================================================
# 7. PREDICTION & RECONSTRUCTION
# =========================================================
y_pred = model.predict(X_test).flatten()

# Get the 'I' values from the original test set for physics reconstruction
# Note: X_test starts from WINDOW_SIZE onwards in the original dataframe
I_test = df['I'].values[split + WINDOW_SIZE:] 
h2_theo = (I_test / 2 / 96500) * 24.4651 * 1000 * 60

h2_actual = df['H2_actual'].values[split + WINDOW_SIZE:]
h2_pred = h2_theo - y_pred

# =========================================================
# 8. METRICS & SAVE
# =========================================================
r2 = r2_score(h2_actual, h2_pred)
rmse = np.sqrt(mean_squared_error(h2_actual, h2_pred))

print(f"✅ LSTM MODEL R2: {r2:.4f}")
print(f"✅ LSTM MODEL RMSE: {rmse:.4f}")

# Save Model and Scaler
model.save("LSTM_model.h5") 
joblib.dump(scaler, "LSTM_scaler.pkl")
joblib.dump(features, "LSTM_features.pkl")

# =========================================================
# 9. UPLOAD TO THINGSPEAK
# =========================================================
upload_url = (
    f"https://api.thingspeak.com/update?"
    f"api_key={ts_write_key}"
    f"&field5={float(r2):.4f}"
    f"&field6={float(rmse):.4f}"
)
try:
    requests.get(upload_url)
    print("📡 Metrics uploaded to ThingSpeak")
except:
    print("⚠️ Upload failed")
