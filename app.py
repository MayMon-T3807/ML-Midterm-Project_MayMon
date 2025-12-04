import streamlit as st
import pandas as pd
import joblib
try:
    model = joblib.load('model_protocol4.pkl')
except:
    import pickle
    with open('model_protocol4.pkl', 'rb') as f:
        model = pickle.load(f)

# App
st.title("✈️ Flight Delay Predictor")

# Only essential inputs
airline = st.selectbox("Airline", ["AA", "DL", "UA", "WN"])
origin = st.selectbox("From", ["ATL", "LAX", "ORD", "DFW"])
destination = st.selectbox("To", ["LAX", "ATL", "ORD", "DFW"])
hour = st.slider("Departure Hour", 5, 22, 12)
distance = st.slider("Distance (miles)", 100, 2000, 500)

# Create simple input
input_data = pd.DataFrame([{
    'ORIGIN_AIRPORT': origin,
    'AIRLINE': airline,
    'DESTINATION_AIRPORT': destination,
    'MONTH': 7,  # Fixed
    'DAY': 15,   # Fixed
    'DAY_OF_WEEK': 3,  # Wednesday
    'SCHEDULED_DEPARTURE': hour * 100,
    'SCHEDULED_ARRIVAL': hour * 100 + 120,  # 2 hour flight
    'SCHEDULED_TIME': 120,
    'DISTANCE': distance,
    'hour_of_day': hour,
    'is_morning_rush': 1 if hour in [6,7,8] else 0,
    'is_evening_rush': 1 if hour in [17,18,19] else 0,
    'is_night_flight': 1 if hour in [22,23,0,1,2,3,4,5] else 0,
    'is_weekend': 0,
    'winter_month': 0,
    'summer_month': 1,
    'holiday_season': 0,
    'is_short_flight': 1 if distance < 500 else 0,
    'is_long_flight': 1 if distance > 2000 else 0
}])

# Predict
if st.button("Check Delay"):
    try:
        result = model.predict(input_data)[0]
        if result == 1:
            st.error("⚠️ Likely to be DELAYED")
        else:
            st.success("✅ Likely to be ON TIME")
    except:
        st.error("Prediction failed")