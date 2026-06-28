# 🎬 CineVision AI – Movie Success Predictor

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)

A production-ready AI/ML application that predicts movie commercial success using machine learning. Built entirely with **Python** and **Streamlit**.

---

## 📋 Project Overview

CineVision AI analyzes movie production features — budget, genre, cast popularity, marketing spend, release timing, and more — to classify films into five success tiers and estimate box office revenue and ROI.

### Prediction Outputs

| Output | Description |
|--------|-------------|
| **Movie Category** | Flop · Average · Hit · Super Hit · Blockbuster |
| **Success Probability** | Confidence-weighted class probability (%) |
| **Predicted Revenue** | Estimated box office gross ($) |
| **ROI** | Return on investment (%) |
| **Confidence Score** | Model certainty for the prediction |

---

## ✨ Features

- 🎯 **5-Class Classification** with probability breakdown
- 💰 **Revenue & ROI Estimation** using feature-calibrated heuristics
- 📊 **Interactive Dashboard** with Plotly gauges, radar, and bar charts
- 🧠 **AI Insights** — automated marketing and release recommendations
- 📄 **PDF Reports** — professional downloadable analysis documents
- 🔍 **Dataset Explorer** — search, filter, statistics, and correlation matrix
- 📈 **Model Performance** — accuracy, confusion matrix, feature importance, ROC curves
- 🎨 **Premium UI** — Netflix-inspired dark theme with glassmorphism

---

## 📁 Folder Structure

```
CineVision-AI/
│
├── app.py                      # Main Streamlit home page
├── train_model.py              # ML training pipeline
├── predict.py                  # Prediction engine
├── generate_assets.py          # Logo & banner generator
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── dataset/
│   └── movies.csv              # Training dataset (auto-generated)
│
├── model/
│   ├── movie_model.pkl         # Best trained classifier
│   ├── encoder.pkl             # Label encoders
│   ├── scaler.pkl              # Feature scaler
│   └── metrics.pkl             # Evaluation metrics
│
├── assets/
│   ├── logo.png                # App logo
│   └── banner.jpg              # Hero banner
│
├── pages/
│   ├── Dashboard.py            # Analytics dashboard
│   ├── Prediction.py           # Movie prediction form
│   ├── Dataset_Explorer.py     # Data exploration
│   ├── Model_Performance.py    # Model evaluation
│   └── About.py                # Project information
│
├── utils/
│   ├── preprocessing.py        # Data cleaning & feature engineering
│   ├── charts.py               # Plotly chart utilities
│   ├── insights.py             # AI insight generation
│   ├── pdf_report.py           # PDF report builder
│   └── helpers.py              # Paths, CSS, formatting
│
└── reports/                    # Generated PDF reports
```

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Steps

```bash
# 1. Navigate to project directory
cd CineVision-AI

# 2. (Optional) Create virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate assets (logo & banner)
python generate_assets.py

# 5. Train the model
python train_model.py

# 6. Launch the application
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 📦 Requirements

| Package | Purpose |
|---------|---------|
| streamlit | Multi-page web application |
| pandas | Data manipulation |
| numpy | Numerical computing |
| scikit-learn | ML models & metrics |
| plotly | Interactive charts |
| matplotlib | Static plotting |
| joblib | Model serialization |
| Pillow | Image asset generation |
| reportlab | PDF report generation |
| xgboost | Gradient boosting (optional) |
| seaborn | Statistical visualizations |

---

## 🧠 Training the Model

```bash
python train_model.py
```

The training pipeline:

1. Loads or creates the movie dataset (TMDB or synthetic)
2. Cleans data — missing values, duplicates, feature engineering
3. Compares **5 algorithms**: Random Forest, Decision Tree, Gradient Boosting, Extra Trees, XGBoost
4. Selects the best model by **F1 Score** (weighted)
5. Saves model artifacts to `model/`

### Sample Training Output

```
Best model: Random Forest (F1: 0.8521)
Accuracy: 85.21% | F1: 0.8521
```

---

## 🖥️ Running the Application

```bash
streamlit run app.py
```

### Pages

| Page | Description |
|------|-------------|
| **Home** | Project overview, features, quick start |
| **Prediction** | Input form → category, revenue, ROI, PDF download |
| **Dashboard** | Gauges, charts, radar, comparison analytics |
| **Dataset Explorer** | Preview, search, filter, statistics, correlations |
| **Model Performance** | Accuracy, confusion matrix, ROC, feature importance |
| **About** | Tech stack, pipeline, use cases |

---

## 📸 Screenshots

> Placeholder — add screenshots after running the app:

| Home | Prediction | Dashboard |
|------|------------|-----------|
| ![Home](docs/screenshots/home.png) | ![Prediction](docs/screenshots/prediction.png) | ![Dashboard](docs/screenshots/dashboard.png) |

---

## 🔮 Future Scope

- [ ] Integration with live TMDB/IMDb API for real-time data
- [ ] Deep learning models (Neural Networks, Transformers)
- [ ] Sentiment analysis from trailer comments and social media
- [ ] Multi-language support for the UI
- [ ] A/B testing for marketing budget optimization
- [ ] Historical trend analysis by decade and region
- [ ] User authentication and saved prediction history
- [ ] Docker containerization for deployment

---

## 📄 License

This project is for educational and demonstration purposes.

---

**CineVision AI © 2026** — Built with ❤️ using Python & Streamlit
