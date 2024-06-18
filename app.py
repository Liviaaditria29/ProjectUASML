import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load data
df = pd.read_csv('euro2024_players.csv')

# Sidebar
st.sidebar.title("Exploratory Analysis")
analysis_type = st.sidebar.selectbox("Select Analysis Type", ["Overview", "Position Distribution", "Age Distribution", "Market Value Distribution"])

if analysis_type == "Overview":
    st.title("Euro 2024 Players Overview")
    st.dataframe(df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Country Distribution")
        country_counts = df['Country'].value_counts()
        st.bar_chart(country_counts)
    with col2:
        st.subheader("Position Distribution")
        position_counts = df['Position'].value_counts()
        st.bar_chart(position_counts)

elif analysis_type == "Position Distribution":
    st.title("Position Distribution")
    fig, ax = plt.figure(), plt.figure().subplots()
    sns.countplot(x='Position', data=df, ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)

elif analysis_type == "Age Distribution":
    st.title("Age Distribution")
    fig, ax = plt.figure(), plt.figure().subplots()
    sns.histplot(df['Age'], bins=20, ax=ax)
    st.pyplot(fig)

elif analysis_type == "Market Value Distribution":
    st.title("Market Value Distribution")
    fig, ax = plt.figure(), plt.figure().subplots()
    sns.histplot(df['MarketValue'], bins=20, ax=ax)
    st.pyplot(fig)
