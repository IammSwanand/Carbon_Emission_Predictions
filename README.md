🌍 Climate Change CO₂ Emission Analysis
A data-driven project to explore and visualize global CO₂ emissions across countries using real-world climate data.


🎯 Project Objectives
Analyze global CO₂ emissions data at the country level


Identify trends, correlations, and patterns among climate-related variables


Visualize emissions data using both static and interactive charts


Build a foundation for future modeling, forecasting, or policy research


🧰 Technologies Used
Tool	Purpose
Python	Core programming language
Pandas	Data manipulation and analysis
Seaborn, Matplotlib	Static data visualization
Plotly Express	Interactive maps and charts
Jupyter / VS Code	Development environment


🗂️ Key Features & Workflow
📥 1. Data Loading & Inspection
Loaded data_cleaned.csv into a DataFrame


Displayed dataset structure using .info() and .head()


Checked for missing/null values and column types


🧹 2. Data Cleaning
Removed rows with missing or non-numeric values in key columns (co2_per_cap, etc.)


📊 3. Exploratory Data Analysis (EDA)
Generated descriptive statistics using .describe()


Plotted a correlation heatmap to visualize relationships between numeric features


Improved readability with annotations, color scaling, and formatting


🧮 4. Multicollinearity Check
Computed Variance Inflation Factor (VIF) to detect multicollinearity


Prepared for future regression modeling or feature selection


🌐 5. Interactive World Map (Choropleth)
Created a Plotly choropleth map using ISO country codes


Filtered for selected countries: IND, USA, PAK, RUS, NZL


Used distinct colors for each country instead of a single gradient


Hover tooltips show CO₂ emissions per capita


Fixed rendering in VS Code by setting:

python
Copy
Edit
import plotly.io as pio  
pio.renderers.default = "browser"
📈 Output & Insights
Heatmaps revealed variable correlations for exploratory insight


Interactive maps allowed visual country-wise comparisons


Dataset is now clean and suitable for:
Machine learning

Time-series forecasting

Policy modeling or reporting
