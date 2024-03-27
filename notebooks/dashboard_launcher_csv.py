import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


df = pd.read_csv('../data/weatherHistory.csv')

df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
df['Year'] = df['Formatted Date'].dt.year
df['Month'] = df['Formatted Date'].dt.month


import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

def plot_weather_condition_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y='Summary', order=df['Summary'].value_counts().index)
    plt.title('Distribution of Weather Conditions')
    plt.xlabel('Count')
    plt.ylabel('Weather Condition')
    st.pyplot(plt)
    plt.clf()  

def plot_avg_temp_by_month(df):
    plt.figure(figsize=(10, 6))
    avg_temp_by_month = df.groupby(['Month'])['Temperature (C)'].mean().reset_index()
    sns.lineplot(data=avg_temp_by_month, x='Month', y='Temperature (C)')
    plt.title('Average Temperature by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Temperature (C)')
    st.pyplot(plt)
    plt.clf()


def plot_conditions_by_month(df):
    conditions_by_month = df.groupby(['Month', 'Summary'])['Formatted Date'].count().unstack(fill_value=0)
    conditions_by_month = conditions_by_month.reset_index()
    conditions_by_month.set_index('Month', inplace=True)
    
    # Plot
    conditions_by_month.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.title('Weather Condition Frequencies by Month')
    plt.xlabel('Month')
    plt.ylabel('Frequency')
    plt.legend(title='Weather Summary', bbox_to_anchor=(1.05, 1), loc='upper left')
    

    plt.tight_layout()
    
    st.pyplot(plt)

def plot_precipitation_frequency(df):
    plt.figure(figsize=(8, 4))
    df['Precip Type'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Precipitation Type Frequency')
    plt.ylabel('')
    st.pyplot(plt)
    plt.clf()

def plot_temp_vs_humidity(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Temperature (C)', y='Humidity')
    plt.title('Temperature vs. Humidity')
    plt.xlabel('Temperature (C)')
    plt.ylabel('Humidity')
    st.pyplot(plt)
    plt.clf()

def plot_wind_speed_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='Wind Speed (km/h)', bins=30, kde=True)
    plt.title('Wind Speed Distribution')
    plt.xlabel('Wind Speed (km/h)')
    plt.ylabel('Frequency')
    st.pyplot(plt)
    plt.clf()
    

def plot_precip_effect_on_temp_humidity(df):
    df_precip = df.dropna(subset=['Precip Type'])
    avg_conditions_by_precip = df_precip.groupby('Precip Type').agg({
        'Temperature (C)': 'mean',
        'Humidity': 'mean'
    }).reset_index()


    categories = avg_conditions_by_precip['Precip Type'].tolist()
    n_categories = len(categories)
    
    temperature = avg_conditions_by_precip['Temperature (C)'].tolist()
    humidity = [x * 100 for x in avg_conditions_by_precip['Humidity'].tolist()]  # Scale humidity to %

    barWidth = 0.4
    r1 = np.arange(n_categories)
    r2 = [x + barWidth for x in r1]

    plt.figure(figsize=(12, 8))

    plt.bar(r1, temperature, color='royalblue', width=barWidth, edgecolor='grey', label='Avg Temp (C)')
    plt.bar(r2, humidity, color='skyblue', width=barWidth, edgecolor='grey', label='Avg Humidity (%)')

    plt.xlabel('Precipitation Type', fontweight='bold')
    plt.xticks([r + barWidth / 2 for r in range(n_categories)], categories)

    plt.legend()
    plt.title('Effect of Precipitation Type on Average Temperature and Humidity')

    st.pyplot(plt)
    plt.clf()  
    
    
def main():
    st.title("Weather Data Insights")

    st.header("Distribution of Weather Conditions")
    plot_weather_condition_distribution(df)

    st.header("Conditions by month")
    plot_conditions_by_month(df)
    
    st.header("Precipitation Type Frequency")
    plot_precipitation_frequency(df)

    st.header("Temperature vs. Humidity")
    plot_temp_vs_humidity(df)

    st.header("Wind Speed Distribution")
    plot_wind_speed_distribution(df)
    
    st.header("Effect of Precipitation Type on Temperature and Humidity")
    plot_precip_effect_on_temp_humidity(df)
    
    st.header("Average Temperature by Month")
    plot_avg_temp_by_month(df)
    

if __name__ == "__main__":
    main()

