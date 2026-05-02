import pandas as pd
import numpy as np
import joblib
import requests
import datetime

import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

# =========================================================
# 1. FETCH 1-MONTH DATA
# =========================================================
def get_one_month_data(channel_id, read_key):
    all_chunks = []
    end_date = datetime.datetime.now()

    for i in range(30):
        chunk_end = end_date - datetime.timedelta(days=i)
        chunk_start = end_date - datetime.timedelta(days=i+1)

        start_str = chunk_start.strftime('%Y-%m-%d%%20%H:%M:%S')
        end_str = chunk_end.strftime('%Y-%m-%d%%20%H:%M:%S')

        url = f'https://api.thingspeak.com/channels/{channel_id}/feeds.csv?api_key={read_key}&start={start_str}&end={end_str}'

        try:
            chunk_df = pd.read_csv(url)
            if not chunk_df.empty:
                all_chunks.append(chunk_df)
        except:
            pass

    return pd.concat(all_chunks).drop_duplicates().reset_index(drop=True)

# =========================================================
# 2. LOAD DATA
# =========================================================
df = get_one_month_data('3321400', '4Q4YD3ZW21602X7L')

df.rename(columns={
    'field1':'V',
    'field2':'I',
    'field3':'T',
    'field4':'H2_actual'
}, inplace=True)

df['created_at'] = pd.to_datetime(df['created_at'])
df = df.sort_values('created_at')

# =========================================================
# 3. RESAMPLE (1 MIN)
# =========================================================
df = df.set_index('created_at')
df = df.resample('1min').mean().dropna().reset_index()

# =========================================================
# 4. PHYSICS + ERROR
# =========================================================
df['theo_H2'] = (df['I'] / 2 / 96500) * 24.4651 * 1000 * 60
df['target_err'] = df['theo_H2'] - df['H2_actual']

# =========================================================
# 5. LAG FEATURES (THE KEY UPGRADE)
# =========================================================

# short-term memory
df['err_lag1'] = df['target_err'].shift(1)
df['err_lag2'] = df['target_err'].shift(2)

# medium memory
df['err_lag5'] = df['target_err'].shift(5)

# long memory (important for trend)
df['err_lag30'] = df['target_err'].shift(30)

# optional smoothing
df['err_ma10'] = df['target_err'].rolling(10).mean()

# =========================================================
# 6. TARGET = FUTURE ERROR
# =========================================================
df['target_t1'] = df['target_err'].shift(-1)

df = df.dropna()

# =========================================================
# 7. FEATURES
# =========================================================
features = [
    'V', 'I', 'T',
    'err_lag1', 'err_lag2',
    'err_lag5', 'err_lag30',
    'err_ma10'
]

X = df[features].values
y = df['target_t1'].values

# =========================================================
# 8. TRAIN / TEST SPLIT
# =========================================================
split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =========================================================
# 9. SCALING
# =========================================================
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# 10. MODEL
# =========================================================
model = MLPRegressor(
    hidden_layer_sizes=(32, 16),
    activation='tanh',
    solver='adam',   # better for temporal patterns
    max_iter=3000,
    random_state=42
)

print("🧠 Training LAG MODEL...")
model.fit(X_train_scaled, y_train)

# =========================================================
# 11. PREDICTION
# =========================================================
y_pred = model.predict(X_test_scaled)

# reconstruct H2
I_test = X_test[:, features.index('I')]
h2_theo = I_test * 7.6

h2_actual = h2_theo - y_test
h2_pred = h2_theo - y_pred

# =========================================================
# 12. METRICS
# =========================================================
r2 = r2_score(h2_actual, h2_pred)
rmse = np.sqrt(mean_squared_error(h2_actual, h2_pred))

print(f"🔥 LAG MODEL R2: {r2:.4f}")
print(f"🔥 LAG MODEL RMSE: {rmse:.4f}")

# =========================================================
# 13. SAVE
# =========================================================
joblib.dump(model, "lag_model.pkl")
joblib.dump(scaler, "lag_scaler.pkl")
joblib.dump(features, "lag_features.pkl")

print("✅ Lag model saved")

# =========================================================
# 14. VISUALIZATION
# =========================================================
plt.figure(figsize=(12,5))
plt.plot(h2_actual[:200], label='Actual')
plt.plot(h2_pred[:200], '--', label='Lag Model')
plt.legend()
plt.title("Lag Model Prediction (t+1)")
plt.show()

# =========================================================
# 15. UPLOAD METRICS TO THINGSPEAK
# =========================================================
print(f"📡 Uploading metrics to ThingSpeak...")

# Using the provided URL structure
upload_url = (
    f"https://api.thingspeak.com/update?"
    f"api_key=F6NHRHZ60PHVSXBP"
    f"&field3={float(r2):.4f}"
    f"&field4={float(rmse):.4f}"
)

try:
    response = requests.get(upload_url)
    if response.status_code == 200:
        print(f"✅ Metrics uploaded successfully: R2={r2:.4f}, RMSE={rmse:.4f}")
    else:
        print(f"⚠️ Upload failed with status code: {response.status_code}")
except Exception as e:
    print(f"❌ Error during upload: {e}")
