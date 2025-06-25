🌍 Climate Change CO₂ Emission Analysis
A data-driven exploration and visualization project focused on understanding the patterns of CO₂ emissions across countries using real-world climate data.

📌 Project Objectives
Analyze global CO₂ emissions data at the country level.

Identify trends, correlations, and patterns in emission-related variables.

Visualize emissions data in meaningful and interactive ways.

Build a strong foundation for future machine learning or policy modeling.

🧰 Tools & Technologies Used
Python – Core programming language

Pandas – Data cleaning and manipulation

Seaborn & Matplotlib – Static data visualizations

Plotly Express – Interactive visualizations and world maps

Jupyter Notebook / VS Code – Development environment

🗂️ Key Steps and Features
1. Data Loading & Inspection
Loaded cleaned climate change dataset (data_cleaned.csv)

Inspected dataset shape, data types, and column overview using data.info() and .head()

2. Data Cleaning
Checked for missing values across all columns

Filtered and cleaned rows with missing or non-numeric values in relevant columns (e.g., co2_per_cap)

3. Exploratory Data Analysis (EDA)
Used .describe() for summary statistics

Created a correlation heatmap using Seaborn for numerical feature relationships

Adjusted heatmap size, label rotation, and annotation formatting for clarity

4. Multicollinearity Analysis (VIF)
Added a reusable function to compute Variance Inflation Factor (VIF)

Prepared the dataset for future regression modeling or dimensionality reduction

5. Interactive World Map
Used Plotly choropleth maps to visualize CO₂ emissions per capita

Initially used color gradients (Reds) for numeric representation

Improved usability by:

Switching to distinct colors per country using color="country"

Displaying selected countries: IND, USA, PAK, RUS, NZL

Customizing tooltips to include actual CO₂ emission values

6. Plotly Rendering Fixes
Resolved rendering issues in VS Code by switching to pio.renderers.default = "browser"

Added HTML export option for browser-based interactivity (optional)

📈 Output & Insights
Clean, readable correlation heatmaps highlighting related metrics

Interactive world map allowing comparison of selected countries' emissions

Dataset prepared for downstream modeling or country-specific emission policy analysis
