import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import streamlit as st

# Membaca data dari file CSV
df = pd.read_csv('euro2024_players.csv')

# Membersihkan dan transformasi data
X = df.drop('Country', axis=1)
y = df['Country']

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membangun model Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Mengevaluasi model
accuracy = rf.score(X_test, y_test)
print(f'Akurasi model: {accuracy:.2f}')

# Implementasi di Streamlit
st.title('Prediksi Negara Pemain Sepak Bola')

st.subheader('Input Data Pemain')
name = st.text_input('Nama', '')
position = st.selectbox('Posisi', X.columns[1])
age = st.number_input('Usia', min_value=16, max_value=40, step=1)
club = st.text_input('Klub', '')
height = st.number_input('Tinggi Badan (cm)', min_value=160, max_value=210, step=1)
foot = st.selectbox('Kaki', ['left', 'right', 'both'])
caps = st.number_input('Jumlah Pertandingan', min_value=0, step=1)
goals = st.number_input('Jumlah Gol', min_value=0, step=1)
market_value = st.number_input('Nilai Pasar (€)', min_value=0, step=1000)

new_player = pd.DataFrame({
    'Name': [name],
    'Position': [position],
    'Age': [age],
    'Club': [club],
    'Height': [height],
    'Foot': [foot],
    'Caps': [caps],
    'Goals': [goals],
    'MarketValue': [market_value]
})

if st.button('Prediksi Negara'):
    prediction = rf.predict(new_player)
    st.write(f'Prediksi: {prediction[0]}')
