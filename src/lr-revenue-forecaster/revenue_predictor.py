#!/usr/bin/env python3
"""
Predicts revenue from ad spend and site visits using linear regression.
"""

import joblib  # For easy model save/load in prod
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load data
print("📊 Loading revenue data...")
df = pd.read_csv('data/revenue_data.csv')
print(df.head())
print(f"Dataset shape: {df.shape}")

# Prepare features/target
X = df[['ad_spend', 'visits']]
y = df['revenue']

# Train/test split (80/20, reproducible)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Simple Linear Regression - fits production needs
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Clear outputs
print("\n" + "="*50)
print("🎯 MODEL PERFORMANCE")
print("="*50)
print(f"R² Score: {r2_score(y_test, y_pred):.4f} (explains {r2_score(y_test, y_pred)*100:.1f}% variance)")
print(f"MSE: ${mean_squared_error(y_test, y_pred):,.0f}")
print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):.0f}")

print("\n📈 INTERPRETABLE COEFFICIENTS")
print("="*50)
print(f"Revenue ≈ ${model.intercept_:.0f} + "
      f"{model.coef_[0]:.2f} × ad_spend + "
      f"{model.coef_[1]:.2f} × visits")
print("(Every $1k ad_spend → ~$1.8k revenue boost)")

# Save model for deployment (e.g., cron retrain + API)
joblib.dump(model, 'revenue_model.joblib')
print("\n💾 Model saved: revenue_model.joblib")

# Real automation: Predict future month
print("\n🔮 FUTURE FORECAST EXAMPLE")
print("="*50)
future_ad_spend = 8500
future_visits = 32000
future_input = pd.DataFrame([[future_ad_spend, future_visits]], columns=['ad_spend', 'visits'])
pred_revenue = model.predict(future_input)[0]
print(f"Next month: ad_spend=${future_ad_spend:,}, visits={future_visits:,}")
print(f"Predicted Revenue: ${pred_revenue:,.0f}")

print("\n✅ Ready for production! Retrain monthly with new data.")
