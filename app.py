import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Tesla Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# =====================================================
# COLORFUL LIGHT THEME (PROFESSIONAL)
# =====================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fb;
    }

    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #4facfe, #00f2fe);
        color: white;
        padding: 20px;
        border-radius: 14px;
        text-align: center;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.12);
    }

    h1, h2, h3 {
        color: #1f2c56;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# =====================================================
# TITLE
# =====================================================
st.markdown(
    """
    <h1 style='text-align:center;'>📈 Tesla Stock Price Prediction</h1>
    <p style='text-align:center; color:gray;'>
    LSTM Deep Learning Model for Time Series Forecasting
    </p>
    """,
    unsafe_allow_html=True
)

# =====================================================
# LOAD DATA
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_csv("TSLA.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

df = load_data()

latest_close = float(df['Adj Close'].iloc[-1])
records = len(df)

# =====================================================
# PREPROCESSING
# =====================================================
data = df[['Adj Close']]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

last_sequence = scaled_data[-60:]

def create_sequences(data, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i-time_steps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)
X = X.reshape(X.shape[0], X.shape[1], 1)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =====================================================
# LSTM MODEL
# =====================================================
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

# =====================================================
# TEST PREDICTION
# =====================================================
predictions = model.predict(X_test, verbose=0)
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# =====================================================
# FUTURE PREDICTION FUNCTION
# =====================================================
def predict_future(model, last_sequence, days):
    future_predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(days):
        pred = model.predict(
            current_sequence.reshape(1, current_sequence.shape[0], 1),
            verbose=0
        )
        future_predictions.append(pred[0, 0])
        current_sequence = np.append(current_sequence[1:], pred[0, 0])

    return scaler.inverse_transform(
        np.array(future_predictions).reshape(-1, 1)
    )

# =====================================================
# DASHBOARD METRIC CARDS
# =====================================================
c1, c2, c3 = st.columns(3)
c1.metric("Latest Close", f"${latest_close:.2f}")
c2.metric("Data Records", records)
c3.metric("Prediction Engine", "RNN / LSTM")

# =====================================================
# TABS (FUTURE FIRST)
# =====================================================
tab1, tab2 = st.tabs(["🔮 Future Prediction", "📊 Analysis & Charts"])

# =====================================================
# TAB 1: FUTURE PREDICTION (PRIMARY)
# =====================================================
with tab1:
    st.subheader("🔮 Predict Future Stock Prices")

    days = st.number_input(
        "Select number of days to predict",
        min_value=1,
        max_value=10,
        value=1,
        step=1
    )

    future_pred = predict_future(model, last_sequence, days)

    st.subheader("📅 Day-wise Price Outlook")

prev_close = latest_close

for i, close_price in enumerate(future_pred.flatten(), start=1):
    open_price = prev_close
    change_pct = ((close_price - open_price) / open_price) * 100

    color = "#2ecc71" if change_pct >= 0 else "#e74c3c"
    arrow = "▲" if change_pct >= 0 else "▼"

    card_html = f"""
    <div style="
        background-color:#ffffff;
        padding:20px;
        border-radius:14px;
        margin-bottom:16px;
        box-shadow:0 4px 12px rgba(0,0,0,0.08);
        border-left:6px solid {color};
        font-family: Arial, sans-serif;
    ">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-size:18px; font-weight:600; color:#1f2c56;">
                Day {i}
            </div>
            <div style="font-size:16px; font-weight:600; color:{color};">
                {arrow} {change_pct:+.2f}%
            </div>
        </div>

        <hr style="margin:12px 0; border:none; border-top:1px solid #eee;">

        <div style="display:flex; justify-content:space-between; font-size:15px;">
            <div>Open</div>
            <div><b>${open_price:.2f}</b></div>
        </div>

        <div style="display:flex; justify-content:space-between; font-size:15px;">
            <div>Close</div>
            <div><b>${close_price:.2f}</b></div>
        </div>
    </div>
    """

    components.html(card_html, height=180)

    prev_close = close_price





# =====================================================
# TAB 2: ANALYSIS & CHARTS
# =====================================================
with tab2:
    st.subheader("📈 Historical Tesla Stock Price")

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df['Adj Close'], linewidth=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("📊 Actual vs Predicted Prices")

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(y_test_actual, label="Actual Price", linewidth=2)
    ax2.plot(predictions, label="Predicted Price", linewidth=2)
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

# =====================================================
# FOOTER
# =====================================================
st.markdown(
    """
    <hr>
    <p style='text-align:center; color:gray;'>
    Developed as a Deep Learning Academic Project using Streamlit
    </p>
    """,
    unsafe_allow_html=True
)
