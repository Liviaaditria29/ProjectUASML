import streamlit as st
import pandas as pd

# Load the data
df = pd.read_csv('/kaggle/input/uefa-euro-2024-players/euro2024_players.csv')

# Set the page title
st.set_page_config(page_title="UEFA Euro 2024 Players")

# Display the data
st.title("UEFA Euro 2024 Players")

try:
    st.dataframe(df)
except Exception as e:
    st.write("An error occurred while displaying the data:")
    st.write(str(e))

# Sidebar filters
st.sidebar.title("Filters")

# Filter by position
position_filter = st.sidebar.multiselect("Select Positions", df['Position'].unique())
if position_filter:
    df = df[df['Position'].isin(position_filter)]

# Filter by country
country_filter = st.sidebar.multiselect("Select Countries", df['Country'].unique())
if country_filter:
    df = df[df['Country'].isin(country_filter)]

# Filter by age
age_min, age_max = st.sidebar.slider("Select Age Range", int(df['Age'].min()), int(df['Age'].max()), (int(df['Age'].min()), int(df['Age'].max())))
df = df[(df['Age'] >= age_min) & (df['Age'] <= age_max)]

# Display the filtered data
st.dataframe(df)
