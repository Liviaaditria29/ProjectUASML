import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the notebook file
with open('Prediksi_pemain_euro_2024_dengan_randomfores_.ipynb', 'r') as f:
    notebook_content = f.read()

# Display the notebook contents in Streamlit
st.title("Prediksi Pemain Euro 2024 dengan Random Forest")
st.markdown(notebook_content)

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('euro2024_players.csv')
    return df

df = load_data()

# Data preview
st.subheader("Data Preview")
st.write(df.sample(10))

# Exploratory Data Analysis
st.subheader("Exploratory Data Analysis")
st.write("Perform your EDA here...")

# Model Training
st.subheader("Model Training")
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15]
}
grid_search = GridSearchCV(clf, params, cv=5)
grid_search.fit(X_train, y_train)

st.write(f"Best parameters: {grid_search.best_params_}")
st.write(f"Best score: {grid_search.best_score_}")

y_pred = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Test Accuracy: {accuracy:.2f}")

# Additional Features
st.subheader("Additional Features")
st.write("Add any additional functionality or interactivity here...")

if __name__ == '__main__':
    st.run()
