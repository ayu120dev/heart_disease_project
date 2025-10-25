import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Memuat dataset
url = "https://storage.googleapis.com/dqlab-dataset/heart_disease.csv"
data = pd.read_csv(url)

# Menampilkan data
st.title('Prediksi Penyakit Jantung')
st.write("Dataset yang digunakan:")
st.dataframe(data)

# Pisahkan fitur (X) dan target (y)
X = data.drop('target', axis=1)
y = data['target']

# Split data menjadi train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat dan latih model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediksi dan akurasi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Akurasi model: {accuracy * 100:.2f}%")

# Menyimpan model
import joblib
joblib.dump(model, 'model_heart_disease.pkl')

# Form input untuk pengguna
st.sidebar.header("Masukkan Data untuk Prediksi")

age = st.sidebar.number_input("Usia", min_value=20, max_value=100, value=50)
sex = st.sidebar.selectbox("Jenis Kelamin", options=["Pria", "Wanita"])
cp = st.sidebar.selectbox("Jenis Nyeri Dada", options=[0, 1, 2, 3])
trestbps = st.sidebar.number_input("Tekanan Darah Saat Istirahat (trestbps)", min_value=90, max_value=200, value=120)
chol = st.sidebar.number_input("Kadar Kolesterol (chol)", min_value=100, max_value=600, value=240)
fbs = st.sidebar.selectbox("Fasting Blood Sugar", options=[0, 1])  # 0 = Tidak, 1 = Ya
restecg = st.sidebar.selectbox("Hasil EKG", options=[0, 1, 2])
thalach = st.sidebar.number_input("Heart Rate Maximum (thalach)", min_value=50, max_value=220, value=150)
exang = st.sidebar.selectbox("Apakah ada angina (nyeri dada akibat aktivitas fisik)?", options=[0, 1])
oldpeak = st.sidebar.number_input("Oldpeak (Depresi ST setelah aktivitas fisik)", min_value=0.0, max_value=6.0, value=1.0)
slope = st.sidebar.selectbox("Slope dari puncak ST (slope)", options=[0, 1, 2])
ca = st.sidebar.selectbox("Jumlah Pembuluh Darah yang Terkendali (ca)", options=[0, 1, 2, 3])
thal = st.sidebar.selectbox("Thalassemia (thal)", options=[1, 2, 3])

# Membuat prediksi untuk input dari sidebar
input_data = [[age, sex == "Pria", cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]

if st.sidebar.button("Prediksi Penyakit Jantung"):
    prediction = model.predict(input_data)
    if prediction == 1:
        st.sidebar.write("Kemungkinan Terkena Penyakit Jantung")
    else:
        st.sidebar.write("Kemungkinan Tidak Terkena Penyakit Jantung")

