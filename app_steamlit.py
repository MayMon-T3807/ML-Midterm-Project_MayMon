# app.py
import streamlit as st
import joblib  # Changed from pickle to joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Load model
@st.cache_resource
def load_model():
    """Load the trained machine learning model"""
    try:
        # Try loading with joblib first (recommended)
        model = joblib.load('flight_delay_model.pkl')
        
        # If it's a pipeline, extract the classifier
        if hasattr(model, 'named_steps'):
            # It's a pipeline, get the classifier
            if 'classifier' in model.named_steps:
                return model.named_steps['classifier']
            else:
                # Return the last step (usually the classifier)
                return model.steps[-1][1]
        else:
            # It's already just the model
            return model
            
    except FileNotFoundError:
        st.error("❌ Model file 'flight_delay_model.pkl' not found!")
        st.info("Please make sure 'flight_delay_model.pkl' is in the same directory as app.py")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Try using the extracted model (not the pipeline)")
        st.stop()


# Load mapping data
@st.cache_data
def load_mapping_data():
    """Load flight data and create mapping dictionaries"""
    try:
        # Load flight data for unique values
        flight = pd.read_csv('flights.csv', nrows=50000)
        flight = flight[
            (flight['ARRIVAL_DELAY'].notna()) &
            (flight['CANCELLED'] == 0) &
            (flight['DIVERTED'] == 0)
        ]
        
        # Load airport and airline mappings
        airports_df = pd.read_csv('airports.csv')
        airlines_df = pd.read_csv('airlines.csv')
        
        # Create mapping dictionaries
        airport_name_mapping = dict(zip(airports_df['IATA_CODE'], airports_df['AIRPORT']))
        airline_name_mapping = dict(zip(airlines_df['IATA_CODE'], airlines_df['AIRLINE']))
        
        # Get unique values from flight data
        unique_airlines = sorted(flight['AIRLINE'].unique())
        unique_origin_airports = sorted(flight['ORIGIN_AIRPORT'].unique())
        unique_destination_airports = sorted(flight['DESTINATION_AIRPORT'].unique())
        
        # Create dropdown options with names
        airline_options = {}
        for code in unique_airlines:
            name = airline_name_mapping.get(code, f"Unknown ({code})")
            airline_options[code] = name
        
        airport_options = {}
        for code in unique_origin_airports:
            name = airport_name_mapping.get(code, f"Unknown ({code})")
            airport_options[code] = name
        
        return {
            'flight': flight,
            'airline_options': airline_options,
            'airport_options': airport_options,
            'unique_airlines': unique_airlines,
            'unique_origin_airports': unique_origin_airports,
            'unique_destination_airports': unique_destination_airports,
            'airline_name_mapping': airline_name_mapping,
            'airport_name_mapping': airport_name_mapping
        }
    except FileNotFoundError as e:
        st.error(f"❌ Required CSV file not found: {str(e)}")
        st.info("Please make sure flights.csv, airports.csv, and airlines.csv are available")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()


# Load label encoders and scaler if they exist
@st.cache_resource
def load_preprocessors():
    """Load label encoders and scaler if they exist"""
    preprocessors = {
        'label_encoders': None,
        'scaler': None
    }
    
    try:
        preprocessors['label_encoders'] = joblib.load('label_encoders.pkl')
    except:
        pass  # Label encoders not found, will skip encoding
    
    try:
        preprocessors['scaler'] = joblib.load('scaler.pkl')
    except:
        pass  # Scaler not found, will skip scaling
    
    return preprocessors


def calculate_derived_features(input_dict):
    """Calculate all derived features from basic inputs"""
    hour_of_day = input_dict['scheduled_departure']
    
    derived = {
        'hour_of_day': hour_of_day,
        'is_morning_rush': 1 if hour_of_day in [6, 7, 8] else 0,
        'is_evening_rush': 1 if hour_of_day in [17, 18, 19] else 0,
        'is_night_flight': 1 if hour_of_day in [22, 23, 0, 1, 2, 3, 4, 5] else 0,
        'is_weekend': 1 if input_dict['day_of_week'] in [6, 7] else 0,
        'winter_month': 1 if input_dict['month'] in [12, 1, 2] else 0,
        'summer_month': 1 if input_dict['month'] in [6, 7, 8] else 0,
        'holiday_season': 1 if input_dict['month'] in [11, 12] else 0,
        'is_short_flight': 1 if input_dict['distance'] < 500 else 0,
        'is_long_flight': 1 if input_dict['distance'] > 2000 else 0
    }
    
    return derived


def preprocess_input(user_data, preprocessors):
    """Apply preprocessing (encoding and scaling) to input data"""
    
    # Make a copy to avoid modifying original
    processed_data = user_data.copy()
    
    # Apply label encoding if available
    if preprocessors['label_encoders'] is not None:
        categorical_cols = ['ORIGIN_AIRPORT', 'AIRLINE', 'DESTINATION_AIRPORT']
        for col in categorical_cols:
            if col in processed_data.columns:
                try:
                    processed_data[col] = preprocessors['label_encoders'][col].transform(
                        processed_data[col].astype(str)
                    )
                except Exception as e:
                    st.warning(f"Could not encode {col}: {str(e)}")
    
    # Apply scaling if available
    if preprocessors['scaler'] is not None:
        numerical_cols = ['MONTH', 'DAY', 'DAY_OF_WEEK', 'SCHEDULED_DEPARTURE', 
                         'SCHEDULED_ARRIVAL', 'SCHEDULED_TIME', 'DISTANCE', 
                         'hour_of_day', 'is_morning_rush', 'is_evening_rush', 
                         'is_night_flight', 'is_weekend', 'winter_month', 
                         'summer_month', 'holiday_season', 'is_short_flight', 
                         'is_long_flight']
        
        # Only scale columns that exist and are not categorical
        cols_to_scale = [col for col in numerical_cols if col in processed_data.columns]
        
        try:
            processed_data[cols_to_scale] = preprocessors['scaler'].transform(
                processed_data[cols_to_scale]
            )
        except Exception as e:
            st.warning(f"Could not scale features: {str(e)}")
    
    return processed_data


def main():
    """Main Streamlit application"""
    # Load model and data
    model = load_model()
    data = load_mapping_data()
    preprocessors = load_preprocessors()
    
    # Streamlit app configuration
    st.set_page_config(
        page_title="Flight Delay Predictor",
        page_icon="✈️",
        layout="wide"
    )
    
    # Title and description
    st.title("✈️ Flight Delay Predictor")
    st.markdown("""
    Predict whether your flight will be delayed by more than 15 minutes based on various factors.
    """)
    
    # Create a sidebar for additional info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This model predicts flight delays using:
        - Airline and airport information
        - Date and time of flight
        - Flight duration and distance
        - Derived features (rush hour, season, etc.)
        
        **Threshold**: Delay > 15 minutes
        """)
        
        st.header("📊 Model Info")
        st.markdown(f"""
        - **Algorithm**: Random Forest Classifier
        - **Model Type**: {type(model).__name__}
        - **Training data**: 50,000+ flight records
        - **Features**: 20 total features
        - **Preprocessing**: SMOTE for class balancing
        """)
        
        # Show preprocessing status
        st.header("🔧 Preprocessing")
        if preprocessors['label_encoders'] is not None:
            st.success("✓ Label encoders loaded")
        else:
            st.info("ℹ️ No label encoders (using raw values)")
        
        if preprocessors['scaler'] is not None:
            st.success("✓ Scaler loaded")
        else:
            st.info("ℹ️ No scaler (using raw values)")
    
    # Main content area
    st.header("Flight Details")
    
    # Create columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🛫 Origin")
        origin_code = st.selectbox(
            "Select Origin Airport",
            options=data['unique_origin_airports'],
            format_func=lambda x: f"{x} - {data['airport_options'].get(x, x)}",
            key="origin_select"
        )
        
        # Show airport info
        if origin_code in data['airport_name_mapping']:
            st.caption(f"**Airport**: {data['airport_name_mapping'][origin_code]}")
    
    with col2:
        st.subheader("🛬 Destination")
        dest_code = st.selectbox(
            "Select Destination Airport",
            options=data['unique_destination_airports'],
            format_func=lambda x: f"{x} - {data['airport_options'].get(x, x)}",
            key="dest_select"
        )
        
        # Show airport info
        if dest_code in data['airport_name_mapping']:
            st.caption(f"**Airport**: {data['airport_name_mapping'][dest_code]}")
    
    with col3:
        st.subheader("✈️ Airline")
        airline_code = st.selectbox(
            "Select Airline",
            options=data['unique_airlines'],
            format_func=lambda x: f"{x} - {data['airline_options'].get(x, x)}",
            key="airline_select"
        )
        
        # Show airline info
        if airline_code in data['airline_name_mapping']:
            st.caption(f"**Airline**: {data['airline_name_mapping'][airline_code]}")
    
    # Divider
    st.divider()
    
    # Flight details in columns
    st.header("📅 Flight Schedule")
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.subheader("Date")
        month = st.selectbox(
            "Month", 
            options=list(range(1, 13)),
            format_func=lambda x: datetime(2024, x, 1).strftime('%B')
        )
        day = st.selectbox("Day", options=list(range(1, 32)))
    
    with col5:
        st.subheader("Time")
        day_of_week = st.selectbox(
            "Day of Week", 
            options=list(range(1, 8)),
            format_func=lambda x: ["Monday", "Tuesday", "Wednesday", 
                                  "Thursday", "Friday", "Saturday", "Sunday"][x-1]
        )
        
        # Scheduled departure in 24-hour format
        scheduled_departure = st.slider(
            "Departure Hour (24-hour format)",
            min_value=0,
            max_value=23,
            value=12,
            help="Hour of departure in 24-hour format (0-23)"
        )
        
        # Format display
        am_pm = "AM" if scheduled_departure < 12 else "PM"
        hour_display = scheduled_departure if scheduled_departure <= 12 else scheduled_departure - 12
        if hour_display == 0:
            hour_display = 12
        st.caption(f"**Departure Time**: {hour_display}:00 {am_pm}")
    
    with col6:
        st.subheader("Flight Info")
        distance = st.slider(
            "Flight Distance (miles)",
            min_value=50,
            max_value=3000,
            value=500,
            step=50,
            help="Distance between origin and destination airports"
        )
        
        scheduled_time = st.slider(
            "Scheduled Flight Time (minutes)",
            min_value=30,
            max_value=600,
            value=120,
            step=15,
            help="Expected flight duration from takeoff to landing"
        )
        
        # Calculate arrival time
        arrival_hour = (scheduled_departure + (scheduled_time // 60)) % 24
        am_pm_arrival = "AM" if arrival_hour < 12 else "PM"
        hour_display_arrival = arrival_hour if arrival_hour <= 12 else arrival_hour - 12
        if hour_display_arrival == 0:
            hour_display_arrival = 12
        st.caption(f"**Estimated Arrival**: {hour_display_arrival}:00 {am_pm_arrival}")
    
    # Divider
    st.divider()
    
    # Show derived features
    st.header("📈 Derived Features")
    
    # Calculate derived features
    input_data = {
        'scheduled_departure': scheduled_departure,
        'day_of_week': day_of_week,
        'month': month,
        'distance': distance
    }
    
    derived_features = calculate_derived_features(input_data)
    
    # Display derived features in a nice way
    col7, col8, col9 = st.columns(3)
    
    with col7:
        rush_hour_status = (
            "Morning" if derived_features['is_morning_rush'] == 1
            else "Evening" if derived_features['is_evening_rush'] == 1
            else "Normal"
        )
        st.metric("Rush Hour", rush_hour_status)
        st.metric("Weekend", "Yes" if derived_features['is_weekend'] == 1 else "No")
    
    with col8:
        season = (
            "Winter" if derived_features['winter_month'] == 1
            else "Summer" if derived_features['summer_month'] == 1
            else "Holiday" if derived_features['holiday_season'] == 1
            else "Regular"
        )
        st.metric("Season", season)
        st.metric("Night Flight", "Yes" if derived_features['is_night_flight'] == 1 else "No")
    
    with col9:
        flight_length = (
            "Short" if derived_features['is_short_flight'] == 1
            else "Long" if derived_features['is_long_flight'] == 1
            else "Medium"
        )
        st.metric("Flight Length", flight_length)
        st.metric("Distance Category", f"{distance} miles")
    
    # Prediction button
    st.divider()
    st.header("🎯 Prediction")
    
    if st.button("Predict Delay Probability", type="primary", use_container_width=True):
        try:
            # Create input DataFrame with all features
            scheduled_arrival_time = (scheduled_departure * 100 + scheduled_time) % 2400
            
            user_data = pd.DataFrame([{
                'ORIGIN_AIRPORT': origin_code,
                'AIRLINE': airline_code,
                'DESTINATION_AIRPORT': dest_code,
                'MONTH': month,
                'DAY': day,
                'DAY_OF_WEEK': day_of_week,
                'SCHEDULED_DEPARTURE': scheduled_departure * 100,  # Convert to HHMM format
                'SCHEDULED_ARRIVAL': scheduled_arrival_time,
                'SCHEDULED_TIME': scheduled_time,
                'DISTANCE': distance,
                'hour_of_day': scheduled_departure,
                'is_morning_rush': derived_features['is_morning_rush'],
                'is_evening_rush': derived_features['is_evening_rush'],
                'is_night_flight': derived_features['is_night_flight'],
                'is_weekend': derived_features['is_weekend'],
                'winter_month': derived_features['winter_month'],
                'summer_month': derived_features['summer_month'],
                'holiday_season': derived_features['holiday_season'],
                'is_short_flight': derived_features['is_short_flight'],
                'is_long_flight': derived_features['is_long_flight']
            }])
            
            # Apply preprocessing if available
            processed_data = preprocess_input(user_data, preprocessors)
            
            # Make prediction
            prediction = model.predict(processed_data)
            probabilities = model.predict_proba(processed_data)
            
            # Display results
            st.subheader("Prediction Results")
            
            # Create columns for results
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                delay_prob = probabilities[0][1] * 100
                on_time_prob = probabilities[0][0] * 100
                
                st.metric("Delay Probability", f"{delay_prob:.1f}%")
                st.metric("On-Time Probability", f"{on_time_prob:.1f}%")
                
                # Show probability bar
                st.progress(delay_prob / 100, text=f"Delay Risk: {delay_prob:.1f}%")
            
            with result_col2:
                if prediction[0] == 1:
                    st.error("⚠️ **FLIGHT LIKELY TO BE DELAYED**")
                    st.markdown("**Prediction**: Delay > 15 minutes")
                    st.markdown(f"**Confidence**: {delay_prob:.1f}%")
                else:
                    st.success("✅ **FLIGHT LIKELY TO BE ON TIME**")
                    st.markdown("**Prediction**: Delay ≤ 15 minutes")
                    st.markdown(f"**Confidence**: {on_time_prob:.1f}%")
            
            # Show detailed breakdown
            with st.expander("View Detailed Analysis"):
                st.markdown("### Feature Analysis")
                
                # Display input features
                st.markdown("#### Input Features:")
                feature_col1, feature_col2 = st.columns(2)
                
                with feature_col1:
                    st.write(f"**Airline**: {data['airline_name_mapping'].get(airline_code, airline_code)}")
                    st.write(f"**Origin**: {data['airport_name_mapping'].get(origin_code, origin_code)}")
                    st.write(f"**Destination**: {data['airport_name_mapping'].get(dest_code, dest_code)}")
                    st.write(f"**Date**: Month {month}, Day {day}")
                    
                with feature_col2:
                    st.write(f"**Departure**: {scheduled_departure}:00")
                    st.write(f"**Flight Time**: {scheduled_time} minutes")
                    st.write(f"**Distance**: {distance} miles")
                    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    st.write(f"**Day of Week**: {day_names[day_of_week-1]}")
                
                # Display derived features
                st.markdown("#### Derived Features:")
                derived_col1, derived_col2 = st.columns(2)
                
                with derived_col1:
                    is_rush_hour = derived_features['is_morning_rush'] == 1 or derived_features['is_evening_rush'] == 1
                    st.write(f"**Rush Hour**: {'Yes' if is_rush_hour else 'No'}")
                    st.write(f"**Weekend**: {'Yes' if derived_features['is_weekend'] == 1 else 'No'}")
                    st.write(f"**Night Flight**: {'Yes' if derived_features['is_night_flight'] == 1 else 'No'}")
                    
                with derived_col2:
                    st.write(f"**Season**: {season}")
                    st.write(f"**Flight Length**: {flight_length}")
                    st.write(f"**Holiday Season**: {'Yes' if derived_features['holiday_season'] == 1 else 'No'}")
            
            # Show recommendations
            st.subheader("📋 Recommendations")
            if prediction[0] == 1:
                st.warning("""
                **Consider these options:**
                - 📅 Book an earlier flight if possible
                - ✈️ Choose a different airline with better on-time performance
                - ⏰ Avoid rush hour flights (6-8 AM or 5-7 PM)
                - 🔄 Allow extra time for connections
                - 💼 Consider travel insurance
                - 📱 Sign up for flight status alerts
                """)
            else:
                st.info("""
                **Good news! Your flight has a high on-time probability.**
                - ✅ Proceed with your travel plans
                - ⏱️ Still allow some buffer time for unexpected delays
                - 📱 Check in online to save time at the airport
                - 🔔 Monitor flight status as departure approaches
                """)
                
        except Exception as e:
            st.error(f"❌ An error occurred during prediction: {str(e)}")
            st.info("Please check that all inputs are valid and try again.")
            
            # Show debug info
            with st.expander("Debug Information"):
                st.write("**Error details:**")
                st.code(str(e))
                st.write("**Model type:**", type(model))
                if hasattr(model, 'feature_names_in_'):
                    st.write("**Expected features:**", model.feature_names_in_)
    
    # Footer
    st.divider()
    st.caption("""
    **Note**: This prediction is based on historical data and machine learning models. 
    Actual flight delays may vary due to unforeseen circumstances such as weather, 
    air traffic control issues, or operational changes by the airline.
    """)
    
    # Add a reset button
    if st.button("🔄 Reset Form", type="secondary"):
        st.rerun()


if __name__ == "__main__":
    main()