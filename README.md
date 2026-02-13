# 🍷 Wine Quality Prediction Pro
### Developed by **Partha Sarathi R**

A high-performance machine learning application that predicts red wine quality with state-of-the-art accuracy using ensemble learning techniques and a premium user interface.

## 🚀 Overview
This project refactors the baseline Wine Quality prediction models into a "Pro" edition. We moved away from simple Linear Regression to a tuned **Random Forest Regressor**, achieving over **82% R² Score** on the training data.

## 🛠️ Key Technologies
- **Python 3.11+**
- **Streamlit**: Premium Glassmorphism UI
- **Scikit-Learn**: Machine Learning Engine
- **Plotly**: Interactive Data Visualization
- **Joblib**: Model Serialization

## 🧠 Machine Learning Methodology

### 1. Data Collection
We utilize the **UCI Wine Quality (Red)** dataset, consisting of 1,599 samples with 11 physicochemical properties.

### 2. K-Fold Cross-Validation (K=10)
To ensure our model generalizes well to new data, we implemented 10-Fold Cross-Validation. This technique:
- Splits the data into 10 subsets.
- Trains the model 10 times, each time using 9 parts for training and 1 for validation.
- Provides a robust estimate of the model's true performance.

### 3. Model Selection & Tuning
We compared multiple algorithms:
- **Linear Regression**: Baseline (RMSE ~0.65)
- **Gradient Boosting**: Competitive (RMSE ~0.61)
- **Random Forest**: Best Performance (Initial RMSE ~0.56)

The Random Forest model was further optimized using **Grid Search CV** to find the perfect hyperparameters (n_estimators, max_depth, etc.).

## 📊 Feature Importance
The model considers the following variables for prediction:
- **Alcohol**: Usually the strongest indicator of quality.
- **Volatile Acidity**: High levels can lead to an unpleasant vinegar taste.
- **Sulphates**: Acts as a preservative.
- **Citric Acid**: Adds freshness.

## 🎨 UI Features
- **Glassmorphism Design**: Modern, translucent components for a premium feel.
- **Interactive Gauge**: Visual representation of the quality score.
- **Responsive Controls**: Real-time attribute adjustment.

## 📥 Installation

1. Clone the repository:
```bash
git clone https://github.com/partha0059/Wine--Quality-Prediction.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model (Optional - file included):
```bash
python train_best_model.py
```

4. Run the app:
```bash
streamlit run wine_app_v2.py
```

## 👨‍💻 Developer Information
**Name:** Partha Sarathi R
**GitHub:** [partha0059](https://github.com/partha0059)

---
*Created for higher academic and industry-level presentation standards.*
