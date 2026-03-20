import streamlit as st
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestRegressor

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="CNC AI Digital Twin", layout="wide")

# =========================
# CUSTOM CSS (SCI-FI UI)
# =========================
st.markdown("""
<style>
body {
    background-color: #0a0f1c;
    color: #00f7ff;
}

h1, h2, h3 {
    color: #00f7ff;
    text-align: center;
}

.blink {
    animation: blinker 1s linear infinite;
    color: red;
    font-weight: bold;
}

@keyframes blinker {
    50% { opacity: 0; }
}

.metric-box {
    background-color: #111827;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #00f7ff;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("🚀 CNC DIGITAL TWIN CONTROL SYSTEM")

# =========================
# CSV UPLOAD
# =========================
uploaded_file = st.sidebar.file_uploader("Upload CNC CSV", type=["csv"])

def generate_data(n=2000):
    data = pd.DataFrame({
        'force': np.random.uniform(100, 500, n),
        'vibration': np.random.uniform(0.1, 1.0, n),
        'temperature': np.random.uniform(30, 120, n),
        'speed': np.random.uniform(1000, 5000, n),
        'feed': np.random.uniform(50, 300, n)
    })

    data['tool_wear'] = (
        0.0005 * data['force'] +
        0.8 * data['vibration'] +
        0.02 * data['temperature'] +
        0.0002 * data['speed'] +
        0.01 * data['feed']
    )
    return data

# =========================
# LOAD DATA
# =========================
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.success("✅ Real dataset loaded")
else:
    data = generate_data()

# =========================
# TRAIN MODEL
# =========================
features = ['force','vibration','temperature','speed','feed']
target = 'tool_wear'

model = RandomForestRegressor(n_estimators=50)
model.fit(data[features], data[target])

# =========================
# LIVE SIMULATION LOOP
# =========================
placeholder = st.empty()

while True:
    # Generate live input
    force = np.random.uniform(100, 500)
    vibration = np.random.uniform(0.1, 1.0)
    temperature = np.random.uniform(30, 120)
    speed = np.random.uniform(1000, 5000)
    feed = np.random.uniform(50, 300)

    input_data = np.array([[force, vibration, temperature, speed, feed]])
    pred = model.predict(input_data)[0]

    # Physics correction
    pred += force * temperature * 0.00001

    # Tool life %
    tool_life = max(0, 100 - pred)

    # G-code
    speed_adj = int(speed * (1 - pred * 0.01))
    feed_adj = int(feed * (1 - pred * 0.01))
    gcode = f"G01 X10 Y10 F{feed_adj} S{speed_adj}"

    # ALERT
    alert = ""
    if tool_life < 30:
        alert = '<p class="blink">⚠ CRITICAL TOOL FAILURE IMMINENT</p>'

    # UI UPDATE
    with placeholder.container():
        col1, col2, col3 = st.columns(3)

        col1.markdown(f"<div class='metric-box'>🔩 Wear<br><h2>{round(pred,2)}</h2></div>", unsafe_allow_html=True)
        col2.markdown(f"<div class='metric-box'>❤️ Tool Life %<br><h2>{round(tool_life,2)}%</h2></div>", unsafe_allow_html=True)
        col3.markdown(f"<div class='metric-box'>🌡 Temp<br><h2>{round(temperature,2)}</h2></div>", unsafe_allow_html=True)

        st.markdown(alert, unsafe_allow_html=True)

        st.subheader("🛠 G-Code Output")
        st.code(gcode)

    time.sleep(1)
