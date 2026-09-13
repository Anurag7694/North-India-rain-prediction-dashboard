# Dashboard for Rain Prediction in North India

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An interactive meteorological analytics and real-time precipitation prediction dashboard for North India. This application combines comprehensive exploratory weather analysis with a production-ready baseline Logistic Regression classifier to predict daily precipitation occurrences across 9 northern states and union territories.

---

## Live Demo Placeholder

> 🌐 **Live Application:** [Deploy on Streamlit Community Cloud](https://share.streamlit.io/) *(Placeholder — replace with your deployed app URL after connecting to Streamlit Cloud)*

---

## Project Objective

Rainfall patterns across North India exhibit strong seasonal and geographical variations, driven by monsoonal dynamics, Himalayan topography, and continental climate influences. The objective of this project is to:

1. **Provide actionable meteorological insights** through historical climate observations (2017–2024) across 9 States and Union Territories (Uttar Pradesh, Rajasthan, Haryana, Punjab, Jammu & Kashmir, Himachal Pradesh, Chandigarh, Uttarakhand, and Delhi).
2. **Deliver real-time precipitation inference** using a scikit-learn Logistic Regression baseline model with automated diurnal temperature range (DTR) and topographic elevation stratification.
3. **Offer an interactive, presentation-ready dashboard** featuring dynamic filtering, synchronized meteorological KPI metrics, and responsive spatial-temporal charts.

---

## Tools Used

- **Programming Language:** Python 3.10+
- **Application Framework:** Streamlit
- **Machine Learning & Pipeline:** Scikit-learn, Joblib
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Interactive Visualizations:** Plotly Express, Plotly Graph Objects

---

## Dataset Description

The dataset comprises **141,589 historical daily station records** collected across weather stations in North India from 2017 through 2024. The data has been rigorously audited, cleaned, and feature-engineered to support both statistical climatology and supervised binary classification.

- **Total Records:** 141,589 daily station observations
- **Rainy Days:** 54,037 days (38.16%) where recorded precipitation > 0 mm
- **No Rain Days:** 87,552 days (61.84%) with 0 mm precipitation (Dry)
- **Geographic Coverage:** 9 States / Union Territories (UP, RJ, HR, PB, JK, HP, CH, UK, DL)

---

## Features

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `avg_temp` | Continuous (Float) | Daily average ambient temperature (°C) |
| `min_temp` | Continuous (Float) | Minimum daily temperature recorded (°C) |
| `max_temp` | Continuous (Float) | Maximum daily temperature recorded (°C) |
| `dtr` | Continuous (Float) | Diurnal Temperature Range: $\text{DTR} = \text{max\_temp} - \text{min\_temp}$ (°C) |
| `wind_speed` | Continuous (Float) | Mean surface wind speed (km/h) |
| `air_pressure` | Continuous (Float) | Surface barometric atmospheric pressure (hPa) |
| `elevation` | Continuous (Integer) | Station altitude above sea level (meters) |
| `elevation_category` | Categorical | Topographical elevation tier (`Low` < 500m, `Medium` 500–1500m, `High` ≥ 1500m) |
| `month` | Categorical | Calendar month of observation (January through December) |
| `season` | Categorical | IMD seasonal classification (`Winter`, `Summer`, `Monsoon`, `Post-monsoon`) |
| `state` | Categorical | State or Union Territory abbreviation (`UP`, `RJ`, `HR`, `PB`, `JK`, `HP`, `CH`, `UK`, `DL`) |
| `rain_flag` *(Target)* | Binary | Precipitation occurrence: `1` = Rain (> 0 mm), `0` = No Rain (0 mm) |

---

## Machine Learning Model

The predictive component utilizes a **Logistic Regression Pipeline** built using `scikit-learn`:

- **Preprocessing Pipeline:**
  - **Numerical Transformer:** `StandardScaler` applied to `['avg_temp', 'min_temp', 'max_temp', 'dtr', 'wind_speed', 'air_pressure', 'elevation']`
  - **Categorical Transformer:** `OneHotEncoder(drop='first', handle_unknown='ignore')` applied to `['elevation_category', 'month', 'season', 'state']`
- **Estimator:** `LogisticRegression(max_iter=1000, random_state=42)`
- **Serialization:** Exported with `joblib` into `models/baseline_logistic_regression.joblib`

---

## Model Metrics

| Metric | Score / Value |
| :--- | :--- |
| **Accuracy** | **79.22%** |
| **Model Type** | Logistic Regression (Baseline Pipeline) |
| **Total Validated Records** | 141,589 observations |
| **Target Distribution** | 38.16% Rain / 61.84% No Rain |

---

## Dashboard Features

1. **Executive KPI Cards:**
   - **Total Records:** Shows current sample count (dynamically updates with filters)
   - **Rainy Days:** Total days with precipitation (> 0 mm) and percentage
   - **No Rain Days:** Total dry days (0 mm) and percentage
   - **Model Accuracy:** Verified baseline model accuracy benchmark (79.22%)

2. **Dynamic Filter Panel:**
   - Multi-dimensional filtering by **State / UT**, **Month**, and **Season**
   - **Reset Filters** button to instantaneously restore all filters to default state

3. **Interactive Real-Time Rain Prediction:**
   - Input controls for temperature, atmospheric pressure, wind speed, elevation, month, season, and state
   - **Automated feature derivation:** Calculates `DTR = max_temp - min_temp` and categorizes elevation dynamically
   - Instant classification card displaying predicted class (`Rain Predicted` / `No Rain Predicted`), confidence probabilities, and a visual progress indicator

4. **Synchronized Meteorological Visualizations:**
   - **Rain vs No Rain Distribution:** Donut chart showing class breakdown for active slice
   - **State-wise Average Rainfall:** Comparative bar chart across states/UTs
   - **Month-wise Average Rainfall:** Chronological bar chart showing precipitation progression across months
   - **Season-wise Average Rainfall:** Distribution across IMD seasons
   - **Year-wise Rainfall Trend:** 2017–2024 historical trend line chart
   - **Top 10 Districts by Average Rainfall:** Fully reactive horizontal bar chart showing the highest rainfall districts for the currently selected state, month, and season filters

5. **Professional Dark Charcoal Theme:**
   - Tailored `#0F172A` slate backdrop with card containers (`#1E293B`) and contrasting cyan (`#38BDF8`) and amber (`#F59E0B`) highlights for optimal readability.

---

## How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/dashboard-for-rain-prediction.git
cd dashboard-for-rain-prediction
```

### 2. Create and Activate a Virtual Environment
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```

The application will open automatically in your default browser at `http://localhost:8501`.

---

## Project Structure

```text
├── .streamlit/
│   └── config.toml                                    # Streamlit theme and server configuration
├── data/
│   └── north_india_weather_feature_engineered.csv     # Feature-engineered weather dataset
├── models/
│   └── baseline_logistic_regression.joblib            # Trained Logistic Regression pipeline
├── app.py                                             # Main Streamlit dashboard application
├── requirements.txt                                   # Python dependencies for deployment
├── .gitignore                                         # Git ignore rules
└── README.md                                          # Project documentation
```

---

## Deployment to Streamlit Community Cloud

1. Push your repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/).
3. Click **New app**.
4. Select your repository, branch (`main`), and set the main file path to `app.py`.
5. Click **Deploy!**
