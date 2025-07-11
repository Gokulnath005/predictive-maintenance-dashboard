import streamlit as st
import pandas as pd
import joblib
import time

# Load the trained ML model
model = joblib.load("edge_model.pkl")

# Define the min and max values used during training for normalization
TEMP_MIN, TEMP_MAX = 60.0, 120.0
RPM_MIN, RPM_MAX = 1000, 3500
TORQUE_MIN, TORQUE_MAX = 3.0, 75.0

# Streamlit page configuration
st.set_page_config(page_title="Monitoring Dashboard", layout="centered")
st.title("Machine Monitoring Dashboard")

# File uploader for sensor dataset
uploaded_file = st.file_uploader("Upload your sensor dataset (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()  # Clean column names

    # Display dataset preview
    st.subheader("Uploaded Dataset Preview")
    st.dataframe(df.head())

    # Placeholder for predictions and results
    prediction_results = []
    time_series = []
    temp_series = []
    rpm_series = []
    torque_series = []

    # Start monitorig
    if st.button("Start Monitoring"):
        st.subheader("Results")
        log_file = open("live_monitor_log.txt", "a")

        for i, row in df.iterrows():
            time.sleep(0.5)  # Simulate real-time stream

            # Extract and safely handle raw inputs
            temp_raw = row.get("Temperature (°C)", TEMP_MIN)
            rpm_raw = row.get("Rotational speed [rpm]", RPM_MIN)
            torque_raw = row.get("Torque [Nm]", TORQUE_MIN)

            temp = temp_raw if pd.notna(temp_raw) else TEMP_MIN
            rpm = rpm_raw if pd.notna(rpm_raw) else RPM_MIN
            torque = torque_raw if pd.notna(torque_raw) else TORQUE_MIN

            # Normalize inputs
            temp_n = (temp - TEMP_MIN) / (TEMP_MAX - TEMP_MIN)
            rpm_n = (rpm - RPM_MIN) / (RPM_MAX - RPM_MIN)
            torque_n = (torque - TORQUE_MIN) / (TORQUE_MAX - TORQUE_MIN)

            # Run prediction
            input_data = [[temp_n, rpm_n, torque_n]]
            prediction = model.predict(input_data)[0]

            # Collect results for charting
            prediction_results.append(prediction)
            time_series.append(i + 1)
            temp_series.append(temp)
            rpm_series.append(rpm)
            torque_series.append(torque)

            # Show alerts
            if prediction == 1:
                st.error(f"Failure detected at row {i+1} — ALERT!")
                
                st.toast("Failure Detected!", icon="⚠️")
                st.markdown(
                    """
                    <div style='background-color:#ffcccc;padding:10px;border-radius:10px;'>
                        <h4 style='color:red;text-align:center;'>⚠️ ACTION REQUIRED</h4>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.success(f"Row {i+1}: Normal operation")

            # Log to file
            log_file.write(f"{time.time()}, {temp:.2f}, {rpm}, {torque}, {prediction}\n")

        log_file.close()
        st.info("Monitoring completed. Logs saved to 'live_monitor_log.txt'")

        # Show log viewer

        # Add dynamic line charts for visualization
        st.subheader("Prediction Timeline")
        result_df = pd.DataFrame({
            "Timestamp (row #)": time_series,
            "Prediction (0=Normal, 1=Failure)": prediction_results
        })
        st.line_chart(result_df.set_index("Timestamp (row #)"))

        st.subheader("Sensor Readings Over Time")
        sensor_df = pd.DataFrame({
            "Timestamp (row #)": time_series,
            "Temperature (°C)": temp_series,
            "Rotational Speed (RPM)": rpm_series,
            "Torque (Nm)": torque_series
        })
        st.line_chart(sensor_df.set_index("Timestamp (row #)"))
