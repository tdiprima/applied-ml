#!/usr/bin/env python3
"""
Scans system metrics (CPU, memory, disk, network) and flags anything unusual.
Uses Isolation Forest — no labels needed, just feed it data.
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest

# Load data
df = pd.read_csv('system_metrics.csv', parse_dates=['timestamp'])
X = df[['cpu_usage', 'memory_usage', 'disk_read_mb', 'disk_write_mb', 'net_packets_sec']]

# Train Isolation Forest (underused automation gem!)
model = IsolationForest(contamination=0.02, random_state=42, n_jobs=-1)
model.fit(X)

# Predict: -1=anomaly, 1=normal
df['is_anomaly'] = model.predict(X)
df['anomaly_score'] = model.decision_function(X)  # Lower = more anomalous

# Outputs: Simple, readable
n_anomalies = (df['is_anomaly'] == -1).sum()
print(f"🚨 ANOMALY REPORT")
print(f"Total samples: {len(df):,}")
print(f"Anomalies detected: {n_anomalies} ({n_anomalies/len(df)*100:.1f}%)")
print("\nTop 10 most anomalous (lowest score):")
print(df[df['is_anomaly'] == -1][['timestamp', 'cpu_usage', 'memory_usage', 
                                   'disk_read_mb', 'disk_write_mb', 'net_packets_sec', 
                                   'anomaly_score']].sort_values('anomaly_score').head(10).round(2))

# Save anomalies to CSV (deploy: email/ alert these)
df[df['is_anomaly'] == -1].to_csv('alert_anomalies.csv', index=False)
print(f"\n💾 Full anomalies saved to 'alert_anomalies.csv'")

# Plot: Time series + anomaly highlights (easy viz for mortals)
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
features = ['cpu_usage', 'memory_usage', 'net_packets_sec']
for i, feat in enumerate(features):
    axes[i].plot(df['timestamp'], df[feat], label=feat, alpha=0.7)
    anomalies = df[df['is_anomaly'] == -1]
    axes[i].scatter(anomalies['timestamp'], anomalies[feat], color='red', s=50, label='Anomaly')
    axes[i].legend()
    axes[i].set_ylabel(feat)
    axes[i].grid(True, alpha=0.3)
axes[-1].set_xlabel('Time')
plt.suptitle('System Metrics: Anomalies Flagged by Isolation Forest 🔍', fontsize=14)
plt.tight_layout()
plt.savefig('anomaly_plot.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Plot saved as 'anomaly_plot.png'")
