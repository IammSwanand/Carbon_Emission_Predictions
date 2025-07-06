import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Load Data
data = pd.read_csv("data/data_cleaned.csv")
selected_countries = ['IND', 'USA', 'PAK', 'RUS', 'NZL']

# Title
st.title("🌍 CO₂ Emissions Analysis Dashboard")

# Sidebar
country = st.sidebar.selectbox("Select a country", sorted(data['country'].dropna().unique()))

# Filtered Data
country_data = data[data['country'] == country]

# Show Stats
import streamlit as st

# Create 3 tabs
tab1, tab2, tab3 = st.tabs(["📊 Correlation Heatmap", "🗺️ CO₂ Emission Map", "📐 VIF Analysis"])

with tab1:
    st.header("📊 Correlation Heatmap")

    country_data = data[data["country"] == country]
    numeric_cols = country_data.select_dtypes(include="number")

    if numeric_cols.shape[0] > 1:
        corr = numeric_cols.corr()

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax
        )
        ax.set_title(f"Correlation Heatmap for {country}")
        st.pyplot(fig)
    else:
        st.warning("Not enough data to generate heatmap for the selected country.")


with tab2:
    st.header("🗺️ CO₂ Emission Map")

    selected_countries = ['IND', 'USA', 'PAK', 'RUS', 'NZL']
    map_data = data[data['country'].isin(selected_countries)]
    map_data = map_data.dropna(subset=['co2_per_cap'])

    fig = px.choropleth(
        map_data,
        locations='country',
        locationmode='ISO-3',
        color='co2_per_cap',
        color_continuous_scale='Viridis',
        hover_name='country',
        title="CO₂ Emissions Per Capita "
    )

    fig.add_scattergeo(
        locations=[country],
        locationmode="ISO-3",
        text=[country],
        marker=dict(color="red", size=10),
        name="Selected"
    )

    st.plotly_chart(fig, use_container_width=True)


with tab3:
    st.header("Multicollinearity Analysis (VIF)")

    # Filter dataset for selected country
    country_data = data[data['country'] == country].dropna()

    # Select only numeric columns
    numeric_df = country_data.select_dtypes(include='number')

    # Avoid calculating VIF on empty or single-row data
    if numeric_df.shape[0] > 1:
        vif_data = pd.DataFrame()
        vif_data["Feature"] = numeric_df.columns
        vif_data["VIF"] = [
            variance_inflation_factor(numeric_df.values, i)
            for i in range(numeric_df.shape[1])
        ]
        st.dataframe(vif_data)
    else:
        st.warning("Not enough data for VIF calculation for selected country.")

