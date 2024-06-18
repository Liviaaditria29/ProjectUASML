import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the notebook file
with open('Prediksi_pemain_euro_2024_dengan_randomfores_.ipynb', 'r') as f:
    notebook_content = f.read()

# Display the notebook contents in Streamlit
st.title("Prediksi Pemain Euro 2024 dengan Random Forest")
st.markdown(notebook_content)

# Add any additional functionality or interactivity here
@st.cache
def load_data():
    df = pd.read_csv('euro2024_players.csv')
    return df

df = load_data()

st.subheader("Data Preview")
st.write(df.sample(10))

# Add any other interactive components, visualizations, or analysis here

if __name__ == '__main__':
    st.run()
