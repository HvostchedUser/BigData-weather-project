import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/weatherHistory.csv")

df['formatted_date'] = pd.to_datetime(df['Formatted Date'])

st.header("Seasonal Variations in Temperature")
fig, ax = plt.subplots()
sns.barplot(x="Season", y="Average Temperature", data=seasonal_data, ax=ax)
st.pyplot(fig)


