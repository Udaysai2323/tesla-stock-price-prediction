import streamlit as st
import pandas as pd
import numpy as np

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Tesla Stock Prediction",
    page_icon="📈",
    layout="wide"
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("Tesla Stock Prediction")
st.sidebar.write("Deep Learning Project")
st.sidebar.write("SimpleRNN & LSTM Models")

# --------------------------------------------------
# Main Title
# --------------------------------------------------

st.title("Tesla Stock Price Prediction")

st.info("""
This project predicts Tesla stock closing prices using Deep Learning.

Models Used:
- SimpleRNN
- LSTM

Forecast Options:
- 1 Day
- 5 Days
- 10 Days
""")

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

try:
    df = pd.read_csv("data/TSLA.csv")
except Exception as e:
    st.error(f"Dataset Error: {e}")
    st.stop()

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

df['Date'] = pd.to_datetime(df['Date'])

close_data = df[['Close']]

# --------------------------------------------------
# Dataset Preview
# --------------------------------------------------

st.subheader("Dataset Preview")

st.dataframe(df)

# --------------------------------------------------
# Dataset Information
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", len(df))

with col2:
    st.metric("Highest Close Price", f"${df['Close'].max():.2f}")

with col3:
    st.metric("Lowest Close Price", f"${df['Close'].min():.2f}")

# --------------------------------------------------
# Tesla Stock Trend
# --------------------------------------------------

st.subheader("Tesla Closing Price Trend")

st.line_chart(
    df.set_index('Date')['Close']
)

# --------------------------------------------------
# Scaling
# --------------------------------------------------

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(close_data)

# --------------------------------------------------
# Load Models
# --------------------------------------------------

try:
    rnn_model = load_model(
        "models/best_rnn_model.h5",
        compile=False
    )

    lstm_model = load_model(
        "models/best_lstm_model.h5",
        compile=False
    )

except Exception as e:
    st.error(f"Model Loading Error: {e}")
    st.stop()

# --------------------------------------------------
# Prediction Function
# --------------------------------------------------

def predict_future(model, data, days):

    temp = list(data[-60:])

    predictions = []

    for _ in range(days):

        x = np.array(temp[-60:])

        x = x.reshape(1, 60, 1)

        pred = model.predict(
            x,
            verbose=0
        )

        temp.append(pred[0][0])

        predictions.append(pred[0][0])

    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    )

    return predictions

# --------------------------------------------------
# User Inputs
# --------------------------------------------------

st.subheader("Prediction Settings")

model_choice = st.selectbox(
    "Select Model",
    ["SimpleRNN", "LSTM"]
)

days = st.selectbox(
    "Select Prediction Period",
    [1, 5, 10]
)

# --------------------------------------------------
# Predict Button
# --------------------------------------------------

if st.button("Predict Future Price"):

    if model_choice == "SimpleRNN":
        model = rnn_model
    else:
        model = lstm_model

    result = predict_future(
        model,
        scaled_data,
        days
    )

    st.success("Prediction Completed Successfully")

    pred_df = pd.DataFrame({
        "Day": np.arange(1, days + 1),
        "Predicted Close Price ($)": result.flatten()
    })

    # ----------------------------------------------
    # Prediction Summary
    # ----------------------------------------------

    st.subheader("Prediction Summary")

    st.metric(
        label="Last Predicted Price",
        value=f"${result[-1][0]:.2f}"
    )

    # ----------------------------------------------
    # Table Output
    # ----------------------------------------------

    st.subheader("Predicted Values")

    st.dataframe(pred_df)

    # ----------------------------------------------
    # Chart Output
    # ----------------------------------------------

    st.subheader("Prediction Graph")

    st.line_chart(
        pred_df.set_index("Day")
    )

    # ----------------------------------------------
    # Download Predictions
    # ----------------------------------------------

    csv = pred_df.to_csv(index=False)

    st.download_button(
        label="Download Predictions CSV",
        data=csv,
        file_name="tesla_predictions.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Tesla Stock Prediction using Deep Learning (SimpleRNN & LSTM)"
)