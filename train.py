import pandas as pd
import numpy as np
import joblib
import requests
import datetime
import time

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

# =========================================================
# 1. CONFIG
# =========================================================
CHANNEL_ID = '3321400'
READ_KEY = '4Q4YD3ZW21602X7L'

# =========================================================
# 2. LOAD DATA (SAFE SINGLE CALL - NO LOOP)
# =========================================================
def get_data():
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.csv?api_key={READ_KEY}&results=10000"
    df = pd.read_csv(url).dropna()
    return df

df = get_data()

# =========================================================
# 3. CLEAN + FORMAT
# =========================================================
df.rename(columns={
    'field1': 'V',
    'field2': 'I',
    'field3': 'T',
    'field4': 'H2_actual'
}, inplace=True)

df['created_at'] = pd.to_datetime(df['created_at'])

df = df.drop_duplicates(subset='created_at')
df = df.sort_values('created_at')

# =========================================================
# 4. RESAMPLE TO 1-MINUTE (SMOOTH SIGNAL)
# =========================================================
df = df.set_index('created_at')
df = df.resample('1min').mean().dropna().reset_index()

print("Final dataset size:", len(df))

# =========================================================
# 5. PHYSICS MODEL (FARADAY LAW)
# =========================================================
F = 96500
Vm = 24.465

df['theo_H2'] = (df['I'] / (2 * F)) * Vm * 1000 * 60

# ERROR TARGET
df['target_err'] = df['theo_H2'] - df['H2_actual']

# =========================================================
# 6. FEATURE ENGINEERING
# =========================================================
df['P'] = df['V'] * df['I']

# Cycle feature (safe)
df['cycle_pos'] = (df['I'] > 0.5).cumsum()

# IMPORTANT FEATURES
features = ['V', 'I', 'T', 'P']

X = df[features].values
y = df['target_err'].values

# =========================================================
# 7. TRAIN / TEST SPLIT (TIME SERIES)
# =========================================================
split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =========================================================
# 8. SCALING
# =========================================================
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================================================
# 9. ANN MODEL
# =========================================================
model = MLPRegressor(
    hidden_layer_sizes=(16, 8),
    activation='tanh',
    solver='lbfgs',
    max_iter=5000,
    random_state=42
)

print("🧠 Training ANN Digital Twin...")
model.fit(X_train_scaled, y_train)

# =========================================================
# 10. PREDICTION
# =========================================================
y_pred_err = model.predict(X_test_scaled)

# =========================================================
# 11. RECONSTRUCT H2 OUTPUT
# =========================================================
I_test = X_test[:, features.index('I')]

h2_theo = (I_test / (2 * F)) * Vm * 1000 * 60
h2_pred = h2_theo - y_pred_err
h2_actual = df['H2_actual'].values[split:]

# Align safety
min_len = min(len(h2_actual), len(h2_pred))
h2_actual = h2_actual[:min_len]
h2_pred = h2_pred[:min_len]

# =========================================================
# 12. METRICS
# =========================================================
r2 = r2_score(h2_actual, h2_pred)
rmse = np.sqrt(mean_squared_error(h2_actual, h2_pred))

print("\n==============================")
print(" DIGITAL TWIN PERFORMANCE")
print("==============================")
print(f"R2   : {r2:.4f}")
print(f"RMSE : {rmse:.4f}")

# =========================================================
# 13. SAVE MODEL + SCALER + FEATURES
# =========================================================
joblib.dump(model, "ann_electrolyser_model.pkl")
joblib.dump(scaler, "ann_scaler.pkl")
joblib.dump(features, "feature_list.pkl")

print("✅ Model, scaler, and features saved successfully")
