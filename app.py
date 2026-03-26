import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# ── Page config ───────────────────────────────────
st.set_page_config(
    page_title="Smart AQI Monitor",
    page_icon="🌬️",
    layout="wide"
)

# ── Load model ────────────────────────────────────
model = joblib.load("aqi_model.pkl")
le    = joblib.load("label_encoder.pkl")

# ── ThingSpeak config ─────────────────────────────
CHANNEL_ID   = os.getenv("CHANNEL_ID")
READ_API_KEY = os.getenv("READ_API_KEY")

# ── AQI colour map ────────────────────────────────
AQI_COLORS = {
    "Good"       : "#2ecc71",
    "Moderate"   : "#f1c40f",
    "Unhealthy"  : "#e67e22",
    "V.Unhealthy": "#e74c3c",
    "HAZARDOUS"  : "#8e44ad"
}

# ── Helper — fetch ThingSpeak data ────────────────
@st.cache_data(ttl=60)
def fetch_thingspeak():
    try:
        url = (f"https://api.thingspeak.com/channels/{CHANNEL_ID}"
               f"/feeds.json?api_key={READ_API_KEY}&results=37")
        r = requests.get(url, timeout=5)
        feeds = r.json()["feeds"]
        df = pd.DataFrame(feeds)
        df = df.rename(columns={
            "field1": "Temperature",
            "field2": "Humidity",
            "field3": "Gas_PPM",
            "field4": "AQI_Label"
        })
        df["created_at"]  = pd.to_datetime(df["created_at"])
        df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
        df["Humidity"]    = pd.to_numeric(df["Humidity"],    errors="coerce")
        df["Gas_PPM"]     = pd.to_numeric(df["Gas_PPM"],     errors="coerce")
        return df.dropna()
    except:
        return None

# ════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════
st.title("🌬️ Smart Air Quality Monitoring System")
st.caption("IoT + ML Dashboard — Arduino + DHT22 + MQ-2 via Wokwi Simulation")
st.divider()

# ════════════════════════════════════════════════
# SECTION 1 — LIVE PREDICTION
# ════════════════════════════════════════════════
st.subheader("Live AQI Prediction")
st.write("Adjust the sliders to match your sensor readings and get an instant ML prediction.")

col1, col2, col3 = st.columns(3)
with col1:
    temperature = st.slider("Temperature (°C)", 0.0, 60.0, 25.0, 0.1)
with col2:
    humidity    = st.slider("Humidity (%)",     0.0, 100.0, 50.0, 0.1)
with col3:
    gas_ppm     = st.slider("Gas PPM",          50.0, 9999.0, 200.0, 1.0)

# Predict
X_input   = np.array([[temperature, humidity, gas_ppm]])
pred      = model.predict(X_input)
pred_prob = model.predict_proba(X_input)
label     = le.inverse_transform(pred)[0]
confidence = round(float(np.max(pred_prob)) * 100, 1)
color      = AQI_COLORS.get(label, "#95a5a6")

# Display prediction result
st.markdown(f"""
<div style="
    background:{color}22;
    border:2px solid {color};
    border-radius:12px;
    padding:20px;
    text-align:center;
    margin:10px 0
">
    <h2 style="color:{color};margin:0">Air Quality: {label}</h2>
    <p style="color:#666;margin:4px 0">
        Model confidence: {confidence}% &nbsp;|&nbsp;
        T={temperature}°C &nbsp; H={humidity}% &nbsp; PPM={gas_ppm}
    </p>
</div>
""", unsafe_allow_html=True)

# Confidence bar chart
prob_df = pd.DataFrame({
    "Category"   : le.classes_,
    "Probability": pred_prob[0] * 100
})
fig_conf = px.bar(
    prob_df, x="Category", y="Probability",
    color="Category",
    color_discrete_map=AQI_COLORS,
    title="Prediction confidence per category (%)",
    labels={"Probability": "Confidence (%)"}
)
fig_conf.update_layout(showlegend=False, height=300)
st.plotly_chart(fig_conf, use_container_width=True)

st.divider()

# ════════════════════════════════════════════════
# SECTION 2 — THINGSPEAK LIVE DATA
# ════════════════════════════════════════════════
st.subheader("Live Sensor Data from ThingSpeak")

df = fetch_thingspeak()

if df is not None and len(df) > 0:

    # Latest reading metric cards
    latest = df.iloc[-1]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Temperature",  f"{latest['Temperature']:.1f} °C")
    m2.metric("Humidity",     f"{latest['Humidity']:.1f} %")
    m3.metric("Gas PPM",      f"{latest['Gas_PPM']:.0f}")
    m4.metric("AQI Label",    str(latest["AQI_Label"]).strip())

    st.write("")

    # Time series charts
    c1, c2 = st.columns(2)

    with c1:
        fig_temp = px.line(
            df, x="created_at", y="Temperature",
            title="Temperature over time",
            labels={"created_at": "Time", "Temperature": "°C"},
            color_discrete_sequence=["#378ADD"]
        )
        fig_temp.update_layout(height=280)
        st.plotly_chart(fig_temp, use_container_width=True)

    with c2:
        fig_hum = px.line(
            df, x="created_at", y="Humidity",
            title="Humidity over time",
            labels={"created_at": "Time", "Humidity": "%"},
            color_discrete_sequence=["#1D9E75"]
        )
        fig_hum.update_layout(height=280)
        st.plotly_chart(fig_hum, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        fig_ppm = px.line(
            df, x="created_at", y="Gas_PPM",
            title="Gas PPM over time",
            labels={"created_at": "Time", "Gas_PPM": "PPM"},
            color_discrete_sequence=["#D85A30"]
        )
        fig_ppm.update_layout(height=280)
        st.plotly_chart(fig_ppm, use_container_width=True)

    with c4:
        label_counts = df["AQI_Label"].str.strip().value_counts()
        fig_pie = px.pie(
            values=label_counts.values,
            names=label_counts.index,
            title="AQI category distribution",
            color=label_counts.index,
            color_discrete_map=AQI_COLORS
        )
        fig_pie.update_layout(height=280)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # Raw data table
    st.subheader("Raw sensor readings")
    st.dataframe(
        df[["created_at","Temperature","Humidity","Gas_PPM","AQI_Label"]]
        .sort_values("created_at", ascending=False)
        .reset_index(drop=True),
        use_container_width=True
    )

else:
    st.warning("Could not fetch ThingSpeak data. "
               "Check your Channel ID and Read API Key above.")

st.divider()

# ════════════════════════════════════════════════
# SECTION 3 — MODEL INFO
# ════════════════════════════════════════════════
st.subheader("ML Model Information")

i1, i2, i3, i4 = st.columns(4)
i1.metric("Model type",   "Random Forest")
i2.metric("Accuracy",     "91.7%")
i3.metric("Training rows","94")
i4.metric("Features",     "Temp, Humidity, PPM")

st.caption(
    "Reference: Mokrani et al. (2019) — Air Quality Monitoring Using IoT: A Survey. "
    "IEEE SmartIoT Conference. DOI: 10.1109/SmartIoT.2019.00028"
)
