import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt 
import numpy as np
import os

# --- MODEL SETUP: REQUIRED FOR MODEL INPUTS ---

ALL_FEATURES = [
    'effective_capacity', 'base_beds', 'occupancy_lag1', 'occupancy_lag7',
    'discharge_rate_per_bed', 'bed_impact_score', 'discharges_lag7',
    'discharges_lag1', 'admission_rate_per_bed_lag7', 'admission_rate_per_bed_lag1',
    'outcome_discharged', 'overflow_lag7', 'overflow_lag1', 'wait_per_triage',
    'avg_wait_minutes_lag1', 'avg_wait_minutes_lag7', 'outcome_transferred', 
    'arrival_source_self', 'sex_F', 'sex_M'
]


def plot_forecast_only(sarimax_forecast):
    """Plots ONLY the new SARIMAX forecast result."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plotting only the SARIMAX Forecast
    ax.plot(sarimax_forecast.index, sarimax_forecast.values, label='Predicted Admissions', color='tab:green', linewidth=3, marker='o', linestyle='--')
    
    # Set titles and labels
    ax.set_title(f"SARIMAX Forecast ({len(sarimax_forecast)} Periods)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Weekly Admissions")
    ax.legend(loc='upper left')
    plt.tight_layout()
    return fig

# --- MAIN APP FUNCTION ---

def run_sarimax_app(model_path):
    
    st.title("🧪 Admissions Forecast Simulator")
    st.write("Target: **Admissions**. Adjust the key variables in the sidebar to simulate future scenarios.")

    # --- 1. Load Model & Data ---
    try:
        # Load the saved model
        with open(model_path, "rb") as f:
            sarimax_model = pickle.load(f)

        # We still need a reference to the last date of historical data 
        # to properly start the forecast, even if we don't plot the history.
        # 🚨 PLACEHOLDER DATA: REPLACE WITH YOUR ACTUAL TEST SERIES 🚨
        date_range_test = pd.date_range(start='2024-08-01', periods=20, freq='W')
        test_data = pd.Series(4200 + 300 * np.sin(np.arange(20)/3), index=date_range_test)
        
    except Exception as e:
        st.error(f"Error loading model or historical data. Check file path and data format. Error: {e}")
        return

    # --- 2. Simulation Controls (Sidebar) ---
    st.sidebar.header("Future Scenario Inputs")
    
    forecast_steps = st.sidebar.number_input(
        "Days/Weeks to Predict",
        min_value=1, max_value=30, value=7
    )

    st.sidebar.subheader("Key Input Variables")

    # Interactive inputs for the most important, controllable variables.
    capacity = st.sidebar.number_input("Effective Capacity", min_value=30, value=150)
    occupancy_lag1 = st.sidebar.slider("Occupancy (Previous Period)", min_value=0.0, max_value=200.0, value=110.0, step=1.0)
    discharge_rate = st.sidebar.slider("Discharge Rate (0.0 to 1.0)", min_value=0.0, max_value=1.0, value=0.35, step=0.01)

    # --- 3. Run Simulation ---
    if st.button('Run Admissions Forecast Simulation'):
        try:
            # Create DataFrame of Exogenous Variables (Future Inputs)
            start_date = test_data.index[-1] + pd.Timedelta(days=1)
            future_dates = pd.date_range(start=start_date, periods=forecast_steps, freq='W') # Adjust freq if your data isn't weekly
            
            future_exog = pd.DataFrame(index=future_dates)
            
            # Populate the DataFrame with interactive inputs
            future_exog['effective_capacity'] = capacity
            future_exog['occupancy_lag1'] = occupancy_lag1
            future_exog['discharge_rate_per_bed'] = discharge_rate
            
            # Populate remaining 17 features with simple mean/placeholder values
            for feature in ALL_FEATURES:
                if feature not in future_exog.columns:
                    # Using a default neutral value
                    future_exog[feature] = 100 if 'score' in feature or 'bed' not in feature else 0.5 
            
            # Ensure columns are in the correct order for the model
            future_exog = future_exog[ALL_FEATURES]

            # Generate the forecast
            with st.spinner('Generating forecast...'):
                forecast_result = sarimax_model.forecast(
                    steps=forecast_steps,
                    exog=future_exog
                )

            st.header("✨ Simulation Results")
            
            # Display the forecast-only plot
            st.subheader("📈 Predicted Admissions")
            forecast_plot = plot_forecast_only(forecast_result)
            st.pyplot(forecast_plot)
            
            # Display numerical data
            results_df = pd.DataFrame(forecast_result, columns=['Predicted Admissions'])
            st.dataframe(results_df)
            st.success("Forecast complete. Adjust inputs and run again for a new scenario.")

        except Exception as e:
            st.error(f"Forecasting Error: Check if the model expects all 20 features and if the column names are correct. Error: {e}")


if __name__ == "__main__":
    MODEL_PATH = r"C:\Users\ASUS PC\Desktop\AMDARI INTERNSHIP\Med_Optix\Optix_repo\Model\sarimax_model.pkl"
    run_sarimax_app(MODEL_PATH)