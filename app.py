import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Judul Aplikasi
st.title('Prediksi Penyakit Jantung')

# URL Dataset
url = "https://storage.googleapis.com/dqlab-dataset/heart_disease.csv"
data = pd.read_csv(url)

# Menampilkan Data
st.write("Dataset:")
st.dataframe(data)

# Pisahkan fitur dan target
X = data.drop('target', axis=1)
y = data['target']

# Split data menjadi training dan test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediksi dan Akurasi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Menampilkan hasil
st.write(f"Akurasi Model: {accuracy * 100:.2f}%")

# Menyimpan model
joblib.dump(model, 'model_heart_disease.pkl')
st.write("Model berhasil disimpan.")