"""
Forecasts daily task volume for the next 30 days using Facebook Prophet.
"""
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet

# Load data (generated separately)
df = pd.read_csv('task_volumes.csv')
df['ds'] = pd.to_datetime(df['ds'])
print("Historical data loaded:", df.shape)
print(df.tail(7))  # Last week for context

# Define holidays (Prophet-aware, affects forecasts)
holidays = pd.DataFrame({
    'holiday': 'christmas',
    'ds': pd.to_datetime([
        '2024-12-25', '2025-12-25'  # Historical + 1 future for forecast
    ]),
    'lower_window': 0,
    'upper_window': 1,
})

# Simple Prophet: auto-seasonality + holidays, flexible trend
model = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True,
    holidays=holidays,
    changepoint_prior_scale=0.05,  # Allows trend changes, stabilizes fast
    seasonality_prior_scale=10.0,  # Smooth seasonality
)
model.fit(df)
print("\nModel fitted & stabilized quickly!")

# Forecast: historical + 30 future days
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Outputs: understandable console + files
print("\nNext 30 days forecast (yhat=predicted volume, CI=confidence):")
fcst_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30).round(0)
print(fcst_table)

# Plot: full view with components
fig = model.plot(forecast)
plt.title("Task Volume Forecast 📈\n(use yhat > 1500 to auto-scale)")
plt.xlabel("Date")
plt.ylabel("Tasks per Day")
plt.savefig('forecast_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Save deployable CSV (cron-job friendly)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv('forecasts.csv', index=False)
print("\n✅ forecasts.csv (next 30+ days)")
print("✅ forecast_plot.png")
print("\nDeployment tip: `yhat.mean() > threshold` → scale up AWS/GCP!")
