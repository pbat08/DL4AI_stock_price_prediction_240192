# 📈 Deep Learning for Stock Market Forecasting 🚀

**Crystal ball? Not quite. But close.** This project explores two major markets the mighty **Nasdaq** and the bustling **Vietnam HOSE** to predict stock prices, spot buy/sell signals, and build an optimized portfolio. From Bidirectional LSTMs to Markowitz mean-variance optimization, it’s a comprehensive deep learning and quantitative finance pipeline.

## 🧐 Project Overview**

We tackle a series of tasks that build from individual price prediction to full portfolio management:

1.  **Nasdaq Next‑Day Prediction**: A Stacked Bi‑LSTM architecture with per‑window scaling achieving high $R^2$ scores.
2.  **Vietnam HOSE Multi‑Horizon Forecasting**: Multi-feature LSTM models for $k$-th day and $k$-consecutive-day price horizons.
3.  **Trading Signal Detection**: Jointly trained Bi‑LSTM classifiers that provide "BUY" or "SELL" probabilities based on technical indicators.
4.  **Portfolio Management Pipeline**: A three-stage framework that ranks stocks, applies a composite risk-screening score, and uses Markowitz optimization.
5.  **Deployment Blueprint**: A production-ready sketch for an API-driven SaaS using FastAPI and Docker.

---

## 📁 Project Structure

```text
.
├── nasdaq/                                      # Nasdaq data & resources
├── task5_deployment/                            # Deployment blueprints (API, SaaS sketch)
├── nasdaq_price_prediction.ipynb                # Task 1: Bi‑LSTM on Nasdaq
├── hose_Vietnam_stock_prediction_(task_2_4).ipynb # Tasks 2–4: HOSE forecasting, signals, portfolio
├── requirements.txt
└── README.md

```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone [https://github.com/pbat08/stock-dl-project.git](https://github.com/pbat08/stock-dl-project.git)
cd stock-dl-project

```

### 2. Install dependencies

```bash
pip install -r requirements.txt

```

### 3. API Setup

For the Vietnam HOSE components, ensure you have access to the `vnstock` library. You may need to register your user session within the notebook:

```python
# Example setup within the notebook
from vnstock import *
# register_user(api_key='YOUR_KEY_HERE')

```

### 4. Run the Notebooks

Open the Jupyter notebooks and execute the cells in order. The notebooks are designed to handle data fetching, preprocessing, training, and visualization sequentially.

---

## 🛠 Main Dependencies

* **Deep Learning**: `tensorflow` (LSTM/Bi-LSTM architectures)
* **Data Acquisition**: `nasdaq`, `vnstock`
* **Data Processing**: `pandas`, `numpy`, `scikit-learn`
* **Optimization & Math**: `scipy` (SLSQP solver for Markowitz)
* **Visualization**: `matplotlib`, `seaborn`

---

## 💡 Key Takeaways

* **The Baseline Challenge**: A naive persistence baseline (Tomorrow = Today) frequently beats LSTMs on single-step MAE. Always benchmark your models against simple statistical approaches.
* **Data Discipline**: Strict temporal splitting and fitting scalers *only* on training data are critical to avoid look-ahead bias and "data leakage."
* **Feature Engineering**: Indicators like RSI and MACD are essential for classification tasks; ablation studies showed an ~8-point F1 gain when these were included.
* **Portfolio Impact**: The Markowitz Max Sharpe strategy delivered an annualized return of 21.66% and a 1.684 Sharpe ratio on the HOSE universe.

---

**Now go predict the unpredictable (with a pinch of salt and a solid baseline).** 📉📈
