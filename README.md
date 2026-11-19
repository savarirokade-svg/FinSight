<<<<<<< HEAD

# FinSight – Credit Default Prediction & Financial Analytics Platform

FinSight is a machine learning-based financial analytics platform that predicts credit default risk and provides interactive insights into customer financial behaviour.

## Project Overview

This project uses historical credit card customer data to train and evaluate machine learning models for predicting whether a customer may default on their next payment.

It also provides an interactive web application built using Flask and Plotly.

## Key Features

- Credit default prediction using Machine Learning.
- Interactive financial analytics dashboard.
- Comparison of four ML algorithms.
- Model performance evaluation.
- Customer-level prediction form.
- Interactive data visualizations.

## Machine Learning Models

Four algorithms were trained and evaluated:

1. Random Forest
2. XGBoost
3. Support Vector Machine (SVM)
4. Gradient Boosting

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.8142 | 0.6497 | 0.3466 | 0.4521 | 0.7626 |
| XGBoost | 0.7658 | 0.4771 | 0.6134 | 0.5368 | 0.7771 |
| SVM | 0.6835 | 0.3724 | 0.6292 | 0.4679 | 0.7096 |
| Gradient Boosting | 0.7673 | 0.4796 | 0.6127 | 0.5381 | 0.7776 |

**Best F1 Score:** Gradient Boosting – 0.5381

## Dataset

- Dataset: Default of Credit Card Clients
- Total records: 30,000
- Input features: 23 original features
- Target: Default payment next month

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Flask
- Plotly
- HTML
- CSS
- Matplotlib
- Seaborn

## Project Structure

```text
FinSight/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── dataset/
├── models/
├── notebooks/
├── reports/
├── templates/
└── static/
    └── css/
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder

```bash
cd FinSight
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment (Windows)

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python app.py
```

### 7. Open in your browser

```text
http://127.0.0.1:5000/
```

## Future Improvements

- Deploy the application online.
- Add more financial risk indicators.
- Improve model performance and probability calibration.
- Add downloadable analytics reports.
