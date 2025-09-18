import streamlit as st
import pandas as pd
import joblib

# Load the trained model
try:
    calibrated_model = joblib.load('iam_risk_model.pkl')
    model_loaded = True
except FileNotFoundError:
    st.error("Error: Model file 'iam_risk_model.pkl' not found. Please train the model first.")
    model_loaded = False

st.title("🔐 AuthX IAM Risk Predictor")
st.write("Upload your authentication log to detect anomalous logins.")

# File uploader
uploaded_file = st.file_uploader("Upload your authentication log (Excel or CSV)", type=["xlsx", "csv"])

def get_location_from_device(device_id):
    """
    Map device ID to a rough geographic location for demo.
    In production, use actual IP geolocation.
    """
    if 'UPNETLT' in str(device_id):
        return (12.9716, 77.5946)  # Bangalore, India
    elif 'CERTIFYDT' in str(device_id):
        return (19.0760, 72.8777)  # Mumbai, India
    else:
        return (0.0, 0.0)  # Unknown/External

def preprocess_and_engineer_features(df):
    """
    Clean, standardize, and engineer features from the authentication log.
    """
    df = df.copy()

    # Rename columns to match our pipeline (adjust column names as per your file)
    # Assuming your uploaded file has columns like 'User', 'TimeStamp', 'Result', etc.
    # If your column names are different, modify this part.
    try:
        df.rename(columns={
            'User': 'user_id',
            'TimeStamp': 'timestamp',
            'Result': 'event_type',
            'AccessDevice': 'device_id',
            'AuthenticationFactor': 'auth_factor'
        }, inplace=True)

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', format='%m/%d/%Y %I:%M:%S %p')

        # Drop rows where timestamp conversion failed
        df.dropna(subset=['timestamp'], inplace=True)

        # Standardize event_type: "Granted" -> "success", "Denied" -> "failure"
        df['event_type'] = df['event_type'].apply(lambda x: 'success' if str(x).lower() == 'granted' else 'failure')

        # Create a combined 'user_agent' for device fingerprinting
        df['user_agent'] = df['device_id'].fillna('Unknown').astype(str) + " | " + df['auth_factor'].fillna('Unknown').astype(str)

        # Handle missing IPs — using device location as proxy
        df['ip_address'] = df['device_id'].apply(lambda x:
            '192.168.1.10' if 'UPNETLT' in str(x) else
            '203.199.123.45' if 'CERTIFYDT' in str(x) else
            '8.8.8.8' # default for unknown/external
        )

        # Sort by user and timestamp
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

        features_list = []
        user_state = {}

        for idx, row in df.iterrows():
            user = row['user_id']
            current_time = row['timestamp']
            current_device = row['user_agent']
            event_type = row['event_type']

            if user not in user_state:
                user_state[user] = {
                    'last_login_time': None,
                    'last_login_location': (0.0, 0.0),
                    'known_devices': set(),
                    'failed_attempts': 0
                }

            state = user_state[user]

            failed_before = state['failed_attempts']
            is_new_device = 1 if current_device not in state['known_devices'] else 0
            is_odd_hour = 1 if 0 <= current_time.hour < 6 else 0

            if state['last_login_time']:
                time_diff_min = (current_time - state['last_login_time']).total_seconds() / 60
                time_since_last = min(time_diff_min, 1440)
            else:
                time_since_last = 1440

            current_location = get_location_from_device(row['device_id'])
            if state['last_login_location'] != (0.0, 0.0):
                 try:
                    distance_km = geodesic(state['last_login_location'], current_location).km
                 except:
                    distance_km = 0.0
            else:
                distance_km = 0.0

            impossible_travel = 1 if (distance_km > 500 and time_since_last < 60) else 0

            feature_row = {
                'original_index': idx,
                'user_id': user,
                'timestamp': current_time,
                'event_type': event_type,
                'device_used': current_device,
                'application': row.get('Application', 'Unknown'), # Use .get() for robustness
                'activity': row.get('Activity', 'Unknown'), # Use .get() for robustness
                'failed_attempts_before': failed_before,
                'is_new_device': is_new_device,
                'is_odd_hour': is_odd_hour,
                'time_since_last_login_minutes': time_since_last,
                'distance_from_last_login_km': distance_km,
                'impossible_travel_flag': impossible_travel
            }
            features_list.append(feature_row)

            if event_type == 'success':
                state['last_login_time'] = current_time
                state['last_login_location'] = current_location
                state['known_devices'].add(current_device)
                state['failed_attempts'] = 0
            else:
                state['failed_attempts'] += 1

        df_features = pd.DataFrame(features_list)

        # Select features for prediction - Ensure columns match model training
        feature_columns = [
            'failed_attempts_before',
            'is_new_device',
            'is_odd_hour',
            'time_since_last_login_minutes',
            'distance_from_last_login_km',
            'impossible_travel_flag'
        ]
        X_predict = df_features[feature_columns]

        return df_features, X_predict

    except KeyError as e:
        st.error(f"Error: Missing expected column in the uploaded file: {e}. Please check your file's column names.")
        return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred during preprocessing or feature engineering: {e}")
        return None, None


def assign_risk_tiers(probabilities):
    tiers = []
    for p in probabilities:
        if p < 0.3:
            tiers.append('Low Risk')
        elif p <= 0.7:
            tiers.append('Medium Risk')
        else:
            tiers.append('High Risk')
    return tiers

if uploaded_file is not None and model_loaded:
    try:
        # Read the uploaded file
        if uploaded_file.name.endswith('.xlsx'):
            df_raw = pd.read_excel(uploaded_file, sheet_name='data')
        elif uploaded_file.name.endswith('.csv'):
            df_raw = pd.read_csv(uploaded_file)
        else:
            st.error("Unsupported file type.")
            df_raw = None

        if df_raw is not None and not df_raw.empty:
            st.write("Raw Data Preview:")
            st.dataframe(df_raw.head())

            # Preprocess and engineer features
            df_features, X_predict = preprocess_and_engineer_features(df_raw)

            if df_features is not None and X_predict is not None:
                st.write("Engineered Features Preview:")
                st.dataframe(df_features[[
                    'user_id', 'timestamp', 'event_type', 'failed_attempts_before',
                    'is_new_device', 'is_odd_hour', 'distance_from_last_login_km', 'impossible_travel_flag'
                ]].head())

                # Predict probabilities
                all_probabilities = calibrated_model.predict_proba(X_predict)[:, 1]
                all_risk_tiers = assign_risk_tiers(all_probabilities)

                # Add to dataframe
                df_final = df_features.copy()
                df_final['anomaly_probability'] = all_probabilities
                df_final['risk_tier'] = all_risk_tiers

                st.write("### Risk Assessment Results:")
                st.dataframe(df_final[[
                    'timestamp', 'user_id', 'event_type', 'application', 'activity',
                    'failed_attempts_before', 'is_new_device', 'is_odd_hour',
                    'anomaly_probability', 'risk_tier'
                ]])

                # Display High Risk Events
                high_risk = df_final[df_final['risk_tier'] == 'High Risk']
                if not high_risk.empty:
                    st.error("🚨 HIGH RISK LOGIN ATTEMPTS:")
                    st.dataframe(high_risk[[
                        'timestamp', 'user_id', 'event_type', 'application', 'activity',
                        'failed_attempts_before', 'is_new_device', 'is_odd_hour',
                        'anomaly_probability', 'risk_tier'
                    ]])
                else:
                    st.success("🎉 No High Risk login attempts detected.")

                # Download link for the report
                csv_report = df_final.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Risk Assessment Report",
                    data=csv_report,
                    file_name='authx_risk_assessment_report.csv',
                    mime='text/csv',
                )

        elif df_raw is not None and df_raw.empty:
            st.warning("Uploaded file is empty.")

    except Exception as e:
        st.error(f"An error occurred while processing the uploaded file: {e}")
