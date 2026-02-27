"""
K-Means for automatic customer segmentation.
Loads customers.csv (5000 realistic records).
Fits K=5 clusters on scaled features (age, income, spending_score, purchases, session_time).
Outputs:
- Cluster sizes & centers (understandable metrics)
- Per-cluster averages
- Automation triggers: Different segments → different workflows
Real automation problem: Trigger emails/campaigns based on cluster.
Simple/deployable: No complex deps, just sklearn/pandas. Run anywhere.
Focus: Interpretable clusters for business rules > perfect accuracy.
"""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load data
print("Loading customer data... 🏪")
df = pd.read_csv('customers.csv')
features = ['age', 'annual_income', 'spending_score', 'num_purchases', 'avg_session_time']
X = df[features].values

# Scale for fair clustering (means=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit K-Means (k=5 as in article, elbow-ish sweet spot)
print("Fitting K-Means (k=5)... 🔍")
model = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = model.fit_predict(X_scaled)
df['cluster'] = clusters

# Outputs: Cluster summary
print("\n📊 CLUSTER SUMMARY")
print("Num customers per cluster:")
print(df['cluster'].value_counts().sort_index())

print("\nCluster centers (scaled back to original units for interpretability):")
centers = scaler.inverse_transform(model.cluster_centers_)
centers_df = pd.DataFrame(centers, columns=features)
print(centers_df.round(2))

# Per-cluster averages
print("\nAverage metrics per cluster:")
cluster_stats = df.groupby('cluster')[features].mean().round(2)
print(cluster_stats)

# Automation triggers: Simple if-rules based on clusters (deployable!)
print("\n🤖 AUTOMATION TRIGGERS (Workflows per segment)")
triggers = {
    0: "👑 Premium whales: High income/spend → VIP invites & upsell emails",
    1: "🛒 Frequent average: Med spend/purchases → Loyalty discounts",
    2: "💤 Inactive low: Low engagement → Win-back campaigns",
    3: "🎯 Young browsers: Low purchases/high time → Product recs",
    4: "🏠 Budget loyal: Low income/med purchases → Budget deals"
}

for c in sorted(df['cluster'].unique()):
    print(f"Cluster {c}: {triggers[c]} ({df['cluster'].value_counts()[c]} customers)")

print("\nReady for deployment! e.g., Integrate with email API using cluster labels. 🚀")
print("Sample for prod: if cluster==0: send_vip_email(customer_id)")
