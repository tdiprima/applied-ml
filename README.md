# Machine Learning Models

A collection of practical, production-minded ML implementations covering the most common real-world use cases — from classification and anomaly detection to forecasting and customer segmentation.

The goal is straightforward: demonstrate that machine learning isn't just academic exercise. Each model here solves a concrete business problem, is implemented cleanly in Python, and is paired with clear documentation explaining the algorithm, the reasoning, and the tradeoffs.

---

## What's Here

The majority of these projects are **classification tasks** — which reflects reality. In applied ML, most business problems boil down to "which bucket does this thing belong to?" The projects span several problem types:

| # | Project | Algorithm | Problem Type |
|---|---------|-----------|-------------|
| 1 | [Support Ticket Auto-Classifier](src/logreg-ticket-classifier/) | Logistic Regression | Classification |
| 2 | [Lead Qualification Model](src/rf-lead-qualification/) | Random Forest | Classification |
| 3 | [Email Spam Filter](src/nb-spam-filter/) | Naive Bayes | Classification |
| 4 | [Customer Segmentation](src/kmeans-customer-segmentation/) | K-Means Clustering | Clustering |
| 5 | [Task Volume Forecaster](src/prophet-task-forecaster/) | Prophet | Time Series Forecasting |
| 6 | [System Anomaly Detector](src/isolation-forest-system-anomaly/) | Isolation Forest | Anomaly Detection |
| 7 | [Revenue Predictor](src/lr-revenue-forecaster/) | Linear Regression | Regression |

---

## Why These Models?

These seven algorithms represent the core toolkit that covers the vast majority of ML problems encountered in practice:

- **Logistic Regression** — The workhorse of binary classification. Interpretable, fast, and often surprisingly hard to beat.
- **Random Forest** — When you need robustness and don't want to spend a week tuning hyperparameters.
- **Naive Bayes** — Deceptively simple, yet the gold standard for text classification and spam filtering.
- **K-Means** — Unsupervised grouping when you don't have labeled data but need structure.
- **Prophet** — Time series forecasting built for messy, real-world data with seasonality and holidays.
- **Isolation Forest** — Anomaly detection without needing labeled examples of "what bad looks like."
- **Linear Regression** — The foundation of predictive modeling; understanding it deeply unlocks everything else.

---

## Documentation

Each project includes a `README.md` with implementation notes. The [`docs/`](docs/) directory contains plain-language write-ups explaining what each algorithm actually does — written to be understood by anyone, not just ML practitioners:

- [docs/o1.md](docs/o1.md) — Logistic Regression explained
- [docs/o2.md](docs/o2.md) — Random Forest explained
- [docs/o3.md](docs/o3.md) — Naive Bayes explained
- [docs/o4.md](docs/o4.md) — K-Means explained
- [docs/o5.md](docs/o5.md) — Prophet explained
- [docs/o6.md](docs/o6.md) — Isolation Forest explained
- [docs/o7.md](docs/o7.md) — Linear Regression explained

<br>
