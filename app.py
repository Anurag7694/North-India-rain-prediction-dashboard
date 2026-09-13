"""
Dashboard for Rain Prediction in North India
=============================================
A professional, presentation-ready Streamlit dashboard for exploratory weather 
analysis and real-time precipitation prediction using a trained Logistic Regression baseline.

Theme: Dark Charcoal / Slate Gray
Layout:
1. Title & Subtitle
2. 4 Executive KPI Cards
3. State, Month, Season Filters + Reset Button
4. Real-Time Rain Prediction Section (independent of filters)
5. 6 Meteorological Charts (3 Rows x 2 Columns):
   Row 1: Rain vs No Rain | State-wise Rainfall
   Row 2: Month-wise Rainfall | Season-wise Rainfall
   Row 3: Year-wise Rainfall Trend | Top 10 Districts (Dynamically filtered)
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------------------
# 1. Streamlit Page Configuration & Dark Charcoal Theme
# -------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard for Rain Prediction in North India",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Dark Charcoal Styling
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    
    /* Header Styles */
    .main-title {
        font-size: 2.25rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #94A3B8;
        margin-bottom: 1.25rem;
    }

    /* KPI Cards */
    .kpi-container {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.1rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 4px;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #64748B;
    }

    /* Section Cards */
    .section-panel {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.3rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Prediction Result Cards */
    .pred-card-rain {
        background: #0C4A6E;
        border: 2px solid #38BDF8;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: #F0F9FF;
    }
    .pred-card-dry {
        background: #78350F;
        border: 2px solid #F59E0B;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        color: #FFFBEB;
    }
    
    /* Form & Input adjustments for dark mode */
    div[data-baseweb="select"] {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 2. File Path Resolution & Data Loading (Relative Paths)
# -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# Primary relative paths
DATA_PATH = Path("Data/north_india_weather_feature_engineered.csv")
MODEL_PATH = Path("models/baseline_logistic_regression.joblib")

# Multi-environment fallback relative to script directory
if not DATA_PATH.exists():
    if (BASE_DIR / "Data" / "north_india_weather_feature_engineered.csv").exists():
        DATA_PATH = BASE_DIR / "Data" / "north_india_weather_feature_engineered.csv"
    elif (BASE_DIR / "cleaned_data" / "north_india_weather_feature_engineered.csv").exists():
        DATA_PATH = BASE_DIR / "cleaned_data" / "north_india_weather_feature_engineered.csv"
    elif (BASE_DIR.parent.parent / "Data" / "north_india_weather_feature_engineered.csv").exists():
        DATA_PATH = BASE_DIR.parent.parent / "Data" / "north_india_weather_feature_engineered.csv"

if not MODEL_PATH.exists():
    if (BASE_DIR / "models" / "baseline_logistic_regression.joblib").exists():
        MODEL_PATH = BASE_DIR / "models" / "baseline_logistic_regression.joblib"
    elif (BASE_DIR.parent.parent / "models" / "baseline_logistic_regression.joblib").exists():
        MODEL_PATH = BASE_DIR.parent.parent / "models" / "baseline_logistic_regression.joblib"

@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        st.error(f"Dataset not found at {DATA_PATH}. Please verify that 'Data/north_india_weather_feature_engineered.csv' exists.")
        st.stop()
    df = pd.read_csv(DATA_PATH)
    df['date_of_record'] = pd.to_datetime(df['date_of_record'], dayfirst=True)
    df['year'] = df['date_of_record'].dt.year
    return df

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}. Please verify that 'models/baseline_logistic_regression.joblib' exists.")
        st.stop()
    return joblib.load(MODEL_PATH)

df_full = load_dataset()
pipeline = load_model()

# -------------------------------------------------------------
# 3. Top Section: Dashboard Title
# -------------------------------------------------------------
st.markdown('<div class="main-title">Dashboard for Rain Prediction in North India</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Meteorological Analytics & Real-Time Precipitation Classification System</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# 4. Filters State Management
# -------------------------------------------------------------
if 'filter_state' not in st.session_state:
    st.session_state.filter_state = "All States"
if 'filter_month' not in st.session_state:
    st.session_state.filter_month = "All Months"
if 'filter_season' not in st.session_state:
    st.session_state.filter_season = "All Seasons"

def reset_filters():
    st.session_state.filter_state = "All States"
    st.session_state.filter_month = "All Months"
    st.session_state.filter_season = "All Seasons"

# -------------------------------------------------------------
# 5. Apply Active Filters to Dataset
# -------------------------------------------------------------
df_filtered = df_full.copy()
if st.session_state.filter_state != "All States":
    df_filtered = df_filtered[df_filtered['state'] == st.session_state.filter_state]
if st.session_state.filter_month != "All Months":
    df_filtered = df_filtered[df_filtered['month'] == st.session_state.filter_month]
if st.session_state.filter_season != "All Seasons":
    df_filtered = df_filtered[df_filtered['season'] == st.session_state.filter_season]

is_filtered = (
    st.session_state.filter_state != "All States" or 
    st.session_state.filter_month != "All Months" or 
    st.session_state.filter_season != "All Seasons"
)

# Compute KPI metrics dynamically based on filtered data
total_records = len(df_filtered)
rainy_days = int((df_filtered['rain_flag'] == 1).sum()) if total_records > 0 else 0
norain_days = int((df_filtered['rain_flag'] == 0).sum()) if total_records > 0 else 0
rainy_pct = (rainy_days / total_records * 100) if total_records > 0 else 0.0
norain_pct = (norain_days / total_records * 100) if total_records > 0 else 0.0
model_accuracy = 79.22

# -------------------------------------------------------------
# 6. Four KPI Cards (Directly below Title)
# -------------------------------------------------------------
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    filter_tag = " (Filtered)" if is_filtered else ""
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-label">Total Records{filter_tag}</div>
        <div class="kpi-value" style="color: #F8FAFC;">{total_records:,}</div>
        <div class="kpi-sub">Daily Station Records</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col2:
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-label">Rainy Days</div>
        <div class="kpi-value" style="color: #38BDF8;">{rainy_days:,}</div>
        <div class="kpi-sub">{rainy_pct:.2f}% of days (Rain &gt; 0 mm)</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col3:
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-label">No Rain Days</div>
        <div class="kpi-value" style="color: #F59E0B;">{norain_days:,}</div>
        <div class="kpi-sub">{norain_pct:.2f}% of days (Dry)</div>
    </div>
    """, unsafe_allow_html=True)

with kpi_col4:
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-label">Model Accuracy</div>
        <div class="kpi-value" style="color: #10B981;">{model_accuracy:.2f}%</div>
        <div class="kpi-sub">Logistic Regression Baseline</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------------------------------------
# 7. State, Month, Season Filters + Reset Button
# -------------------------------------------------------------
with st.container():
    st.markdown('<div class="section-panel" style="padding: 1rem 1.2rem; margin-bottom: 1.2rem;">', unsafe_allow_html=True)
    st.markdown('<span style="font-weight:600; color:#CBD5E1; font-size:0.95rem;">Filter Meteorological Data:</span>', unsafe_allow_html=True)
    
    f_c1, f_c2, f_c3, f_c4 = st.columns([1.1, 1.1, 1.1, 0.7])
    
    with f_c1:
        state_list = ["All States"] + sorted(df_full['state'].unique().tolist())
        st.selectbox(
            "State / UT",
            options=state_list,
            key="filter_state"
        )
    with f_c2:
        months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                        'July', 'August', 'September', 'October', 'November', 'December']
        month_list = ["All Months"] + months_order
        st.selectbox(
            "Month",
            options=month_list,
            key="filter_month"
        )
    with f_c3:
        season_list = ["All Seasons", "Winter", "Summer", "Monsoon", "Post-monsoon"]
        st.selectbox(
            "Season",
            options=season_list,
            key="filter_season"
        )
    with f_c4:
        st.write("&nbsp;")
        st.button("Reset Filters", on_click=reset_filters, use_container_width=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# 8. Rain Prediction Section (Independent of filters)
# -------------------------------------------------------------
with st.container():
    st.markdown('<div class="section-panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🔮 Real-Time Rain Prediction Section</div>', unsafe_allow_html=True)
    
    p_left, p_right = st.columns([1.15, 0.85])
    
    with p_left:
        st.markdown("<span style='font-size:0.88rem; font-weight:600; color:#94A3B8;'>Atmospheric & Temperature Inputs:</span>", unsafe_allow_html=True)
        i_c1, i_c2, i_c3 = st.columns(3)
        with i_c1:
            avg_temp = st.number_input("Average Temp (°C)", min_value=-15.0, max_value=50.0, value=28.0, step=0.5)
        with i_c2:
            min_temp = st.number_input("Minimum Temp (°C)", min_value=-20.0, max_value=40.0, value=24.0, step=0.5)
        with i_c3:
            max_temp = st.number_input("Maximum Temp (°C)", min_value=-10.0, max_value=55.0, value=33.5, step=0.5)
            
        i_c4, i_c5, i_c6 = st.columns(3)
        with i_c4:
            wind_speed = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=80.0, value=11.2, step=0.5)
        with i_c5:
            air_pressure = st.number_input("Air Pressure (hPa)", min_value=900.0, max_value=1050.0, value=997.5, step=0.5)
        with i_c6:
            elevation = st.number_input("Elevation (meters)", min_value=0, max_value=4500, value=168, step=10)
            
        st.markdown("<span style='font-size:0.88rem; font-weight:600; color:#94A3B8;'>Spatio-Temporal Inputs:</span>", unsafe_allow_html=True)
        i_c7, i_c8, i_c9 = st.columns(3)
        with i_c7:
            pred_month = st.selectbox("Prediction Month", options=months_order, index=6)  # July
        with i_c8:
            seasons_catalog = ['Winter', 'Summer', 'Monsoon', 'Post-monsoon']
            default_sea_idx = 2 if pred_month in ['June', 'July', 'August', 'September'] else (
                0 if pred_month in ['December', 'January', 'February'] else (
                    1 if pred_month in ['March', 'April', 'May'] else 3
                )
            )
            pred_season = st.selectbox("Prediction Season", options=seasons_catalog, index=default_sea_idx)
        with i_c9:
            pred_states = sorted(df_full['state'].unique().tolist())
            pred_state = st.selectbox("Prediction State", options=pred_states, index=pred_states.index('UP') if 'UP' in pred_states else 0)

        # Automatic Calculations: DTR and Elevation Category
        dtr = round(max_temp - min_temp, 2)
        if elevation < 500:
            elev_cat = "Low"
        elif elevation < 1500:
            elev_cat = "Medium"
        else:
            elev_cat = "High"

        st.markdown(f"""
        <div style="background:#0F172A; border:1px solid #334155; border-radius:8px; padding:10px 14px; margin-top:8px; display:flex; justify-content:space-between; font-size:0.88rem;">
            <span>💡 <strong>Calculated DTR:</strong> <span style="color:#38BDF8; font-weight:700;">{dtr}°C</span></span>
            <span>⛰️ <strong>Elevation Tier:</strong> <span style="color:#38BDF8; font-weight:700;">{elev_cat}</span> (&lt;500m Low, 500–1500m Med, ≥1500m High)</span>
        </div>
        """, unsafe_allow_html=True)

    with p_right:
        st.markdown("<span style='font-size:0.88rem; font-weight:600; color:#94A3B8;'>Prediction Result:</span>", unsafe_allow_html=True)
        
        # Prepare feature vector exactly matching training pipeline
        input_row = pd.DataFrame([{
            'avg_temp': avg_temp,
            'min_temp': min_temp,
            'max_temp': max_temp,
            'dtr': dtr,
            'wind_speed': wind_speed,
            'air_pressure': air_pressure,
            'elevation': elevation,
            'elevation_category': elev_cat,
            'month': pred_month,
            'season': pred_season,
            'state': pred_state
        }])
        
        # Inference using loaded Pipeline
        pred_label = pipeline.predict(input_row)[0]
        pred_probs = pipeline.predict_proba(input_row)[0]
        prob_dry = pred_probs[0] * 100
        prob_rain = pred_probs[1] * 100
        
        if pred_label == 1:
            st.markdown(f"""
            <div class="pred-card-rain">
                <div style="font-size: 2.8rem; line-height: 1;">🌧️</div>
                <h2 style="color: #38BDF8; margin: 0.4rem 0 0.2rem 0; font-size: 1.6rem;">Rain Predicted</h2>
                <p style="color: #BAE6FD; font-size: 0.92rem; margin-bottom: 0.8rem;">
                    Atmospheric moisture and seasonal indicators favor rain.
                </p>
                <div style="background: rgba(15, 23, 42, 0.6); border-radius: 8px; padding: 8px 12px; margin-top: 4px;">
                    <div style="display: flex; justify-content: space-between; font-weight: 600; font-size: 0.95rem;">
                        <span style="color:#E2E8F0;">Rain Probability:</span>
                        <span style="color: #38BDF8;">{prob_rain:.1f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="pred-card-dry">
                <div style="font-size: 2.8rem; line-height: 1;">☀️</div>
                <h2 style="color: #FBBF24; margin: 0.4rem 0 0.2rem 0; font-size: 1.6rem;">No Rain Predicted</h2>
                <p style="color: #FDE68A; font-size: 0.92rem; margin-bottom: 0.8rem;">
                    Atmospheric conditions indicate clear, dry weather.
                </p>
                <div style="background: rgba(15, 23, 42, 0.6); border-radius: 8px; padding: 8px 12px; margin-top: 4px;">
                    <div style="display: flex; justify-content: space-between; font-weight: 600; font-size: 0.95rem;">
                        <span style="color:#E2E8F0;">Dry Probability:</span>
                        <span style="color: #FBBF24;">{prob_dry:.1f}%</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        col_pb1, col_pb2 = st.columns(2)
        col_pb1.metric("Dry Probability", f"{prob_dry:.1f}%")
        col_pb2.metric("Rain Probability", f"{prob_rain:.1f}%")
        st.progress(float(pred_probs[1]))

    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# 9. Six Meteorological Charts (3 Rows x 2 Columns)
# -------------------------------------------------------------
st.markdown('<div class="section-header">📊 Meteorological Charts & Analytics</div>', unsafe_allow_html=True)

# Helper function for dark Plotly chart styling
def style_dark_chart(fig, height=310):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1E293B",
        plot_bgcolor="#1E293B",
        margin=dict(t=30, b=30, l=40, r=20),
        height=height,
        font=dict(color="#F1F5F9", family="Inter, sans-serif"),
        xaxis=dict(gridcolor="#334155", zeroline=False),
        yaxis=dict(gridcolor="#334155", zeroline=False)
    )
    return fig

if len(df_filtered) == 0:
    st.warning("⚠️ No records match the selected filter combination. Please reset or choose different filters.")
else:
    # ---------------------------------------------------------
    # ROW 1: Rain vs No Rain | State-wise Rainfall
    # ---------------------------------------------------------
    r1_col1, r1_col2 = st.columns(2)
    
    with r1_col1:
        st.markdown('<div class="section-panel" style="padding:1rem;">', unsafe_allow_html=True)
        st.markdown("<div style='font-weight:600; color:#F8FAFC; margin-bottom:4px;'>1. Rain vs No Rain Distribution</div>", unsafe_allow_html=True)
        st.caption("Class proportions across selected filtered records")
        
        dist_df = pd.DataFrame({
            'Category': ['No Rain (0)', 'Rain (1)'],
            'Days': [norain_days, rainy_days]
        })
        fig1 = px.pie(
            dist_df,
            names='Category',
            values='Days',
            color='Category',
            color_discrete_map={'No Rain (0)': '#F59E0B', 'Rain (1)': '#38BDF8'},
            hole=0.45
        )
        fig1.update_traces(textposition='inside', textinfo='percent+label+value')
        style_dark_chart(fig1)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r1_col2:
        st.markdown('<div class="section-panel" style="padding:1rem;">', unsafe_allow_html=True)
        st.markdown("<div style='font-weight:600; color:#F8FAFC; margin-bottom:4px;'>2. State-wise Average Rainfall (mm/day)</div>", unsafe_allow_html=True)
        st.caption("Daily rainfall mean comparison by State / UT")
        
        state_rain = df_filtered.groupby('state')['rainfall'].mean().reset_index().sort_values('rainfall', ascending=False)
        fig2 = px.bar(
            state_rain,
            x='state',
            y='rainfall',
            color='rainfall',
            color_continuous_scale=['#0C4A6E', '#0284C7', '#38BDF8'],
            labels={'state': 'State', 'rainfall': 'Avg Rain (mm)'},
            text=state_rain['rainfall'].round(2)
        )
        fig2.update_traces(textposition='outside')
        fig2.update_layout(coloraxis_showscale=False)
        style_dark_chart(fig2)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------
    # ROW 2: Month-wise Rainfall | Season-wise Rainfall
    # ---------------------------------------------------------
    r2_col1, r2_col2 = st.columns(2)
    
    with r2_col1:
        st.markdown('<div class="section-panel" style="padding:1rem;">', unsafe_allow_html=True)
        st.markdown("<div style='font-weight:600; color:#F8FAFC; margin-bottom:4px;'>3. Month-wise Average Rainfall (mm/day)</div>", unsafe_allow_html=True)
        st.caption("Chronological progression across months")
        
        avail_months = [m for m in months_order if m in df_filtered['month'].unique()]
        month_rain = df_filtered.groupby('month')['rainfall'].mean().reindex(avail_months).reset_index()
        fig3 = px.bar(
            month_rain,
            x='month',
            y='rainfall',
            labels={'month': 'Month', 'rainfall': 'Avg Rain (mm)'},
            color='rainfall',
            color_continuous_scale=['#115E59', '#0D9488', '#2DD4BF'],
            text=month_rain['rainfall'].round(2)
        )
        fig3.update_traces(textposition='outside')
        fig3.update_layout(coloraxis_showscale=False)
        style_dark_chart(fig3)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r2_col2:
        st.markdown('<div class="section-panel" style="padding:1rem;">', unsafe_allow_html=True)
        st.markdown("<div style='font-weight:600; color:#F8FAFC; margin-bottom:4px;'>4. Season-wise Average Rainfall (mm/day)</div>", unsafe_allow_html=True)
        st.caption("Precipitation volume by IMD seasonal categories")
        
        season_cat = ['Winter', 'Summer', 'Monsoon', 'Post-monsoon']
        avail_seasons = [s for s in season_cat if s in df_filtered['season'].unique()]
        season_rain = df_filtered.groupby('season')['rainfall'].mean().reindex(avail_seasons).reset_index()
        fig4 = px.bar(
            season_rain,
            x='season',
            y='rainfall',
            color='season',
            color_discrete_sequence=['#93C5FD', '#FCD34D', '#38BDF8', '#FB923C'],
            labels={'season': 'Season', 'rainfall': 'Avg Rain (mm)'},
            text=season_rain['rainfall'].round(2)
        )
        fig4.update_traces(textposition='outside')
        fig4.update_layout(showlegend=False)
        style_dark_chart(fig4)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------------------------------------------
    # ROW 3: Year-wise Rainfall Trend | Top 10 Districts
    # ---------------------------------------------------------
    r3_col1, r3_col2 = st.columns(2)
    
    with r3_col1:
        st.markdown('<div class="section-panel" style="padding:1rem;">', unsafe_allow_html=True)
        st.markdown("<div style='font-weight:600; color:#F8FAFC; margin-bottom:4px;'>5. Year-wise Rainfall Trend (mm/day)</div>", unsafe_allow_html=True)
        st.caption("Historical daily precipitation average by year (2017–2024)")
        
        year_rain = df_filtered.groupby('year')['rainfall'].mean().reset_index()
        fig5 = px.line(
            year_rain,
            x='year',
            y='rainfall',
            markers=True,
            labels={'year': 'Year', 'rainfall': 'Avg Rain (mm)'}
        )
        fig5.update_traces(line_color='#38BDF8', line_width=3, marker=dict(size=8, color='#0284C7'))
        style_dark_chart(fig5)
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with r3_col2:
        st.markdown('<div class="section-panel" style="padding:1rem;">', unsafe_allow_html=True)
        st.markdown("<div style='font-weight:600; color:#F8FAFC; margin-bottom:4px;'>6. Top 10 Districts by Average Rainfall (mm/day)</div>", unsafe_allow_html=True)
        
        # Dynamic label based on active filters (Must update dynamically according to filters)
        active_filter_desc = []
        if st.session_state.filter_state != "All States":
            active_filter_desc.append(f"State: {st.session_state.filter_state}")
        if st.session_state.filter_month != "All Months":
            active_filter_desc.append(f"Month: {st.session_state.filter_month}")
        if st.session_state.filter_season != "All Seasons":
            active_filter_desc.append(f"Season: {st.session_state.filter_season}")
        filter_subtitle = " | ".join(active_filter_desc) if active_filter_desc else "All North India"
        st.caption(f"Dynamically calculated for: **{filter_subtitle}**")
        
        # Group by district and state within df_filtered
        dist_rain = df_filtered.groupby(['district', 'state'])['rainfall'].mean().reset_index()
        dist_rain['district_label'] = dist_rain['district'] + " (" + dist_rain['state'] + ")"
        top10_dist = dist_rain.sort_values('rainfall', ascending=True).tail(10)
        
        fig6 = px.bar(
            top10_dist,
            y='district_label',
            x='rainfall',
            orientation='h',
            labels={'district_label': 'District', 'rainfall': 'Avg Rain (mm)'},
            color='rainfall',
            color_continuous_scale=['#4C1D95', '#7C3AED', '#A78BFA'],
            text=top10_dist['rainfall'].round(2)
        )
        fig6.update_traces(textposition='outside')
        fig6.update_layout(coloraxis_showscale=False)
        style_dark_chart(fig6)
        st.plotly_chart(fig6, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# 10. Clean Footer
# -------------------------------------------------------------
st.markdown("<br><hr style='border-color: #334155;'>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; color: #64748B; font-size: 0.85rem; padding-bottom: 24px;'>"
    "Dashboard for Rain Prediction in North India • Baseline Logistic Regression Model • Cleaned Historical Weather Dataset"
    "</div>", 
    unsafe_allow_html=True
)
