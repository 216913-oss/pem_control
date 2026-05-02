import pandas as pd
import numpy as np
import joblib
import requests
import datetime
import time
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler

# --- CONFIG ---
CHANNEL_ID = '3321400'
READ_KEY = '4Q4YD3ZW21602X7L'

def get_data():
    all_chunks = []
    end_date = datetime.datetime.now()
    for i in range(30): # 30 days
        chunk_start = end_date - datetime.timedelta(days=i+1)
        chunk_end = end_date - datetime.timedelta(days=i)
        url = f'https://api.thingspeak.com/channels/{CHANNEL_ID}/feeds.csv?api_key={READ_KEY}&start={chunk_start.strftime("%Y-%m-%d%%20%H:%M:%S")}&end={chunk_end.strftime("%Y-%m-%d%%20%H:%M:%S")}'
        try:
            df = pd.read_csv(url)
            if not df.empty: all_chunks.append(df)
            time.sleep(0.2)
        except: pass
    return pd.concat(all_chunks).drop_duplicates()

# 1. Fetch & Clean
df = get_data()
df.rename(columns={'field1':'V', 'field2':'I', 'field3':'T', 'field4':'H2_actual'}, inplace=True)
df['created_at'] = pd.to_datetime(df['created_at'])
df = df.set_index('created_at').resample('1min').mean().dropna().reset_index()

# 2. Features
df['power'] = df['V'] * df['I']
X = df[['V', 'I', 'T']].values
y = (df['I'] * 7.6) - df['H2_actual'] # target_err

# 3. Scale & Train
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
model = MLPRegressor(hidden_layer_sizes=(16, 8), activation='tanh', solver='lbfgs', max_iter=5000, random_state=42)
model.fit(X_scaled, y)

# 4. Save (No absolute paths!)
joblib.dump(model, "ann_electrolyser_model.pkl")
joblib.dump(scaler, "ann_scaler.pkl")
print("✅ Training Complete")
