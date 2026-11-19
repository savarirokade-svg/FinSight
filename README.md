# FinSight – Credit Default Prediction & Financial Analytics Platform

An intelligent Machine Learning platform that predicts credit default risk using customer financial information and multiple classification algorithms.

## About the Project

FinSight is a Machine Learning-based financial analytics platform designed to predict whether a credit card customer is likely to default on their payment next month.

The system analyzes customer information, payment history, credit limits, and bill amounts to generate predictions and support financial risk analysis.

## Objectives

- Predict customer credit default risk.
- Analyze customer financial information.
- Compare different Machine Learning algorithms.
- Evaluate model performance using classification metrics.
- Provide an interactive platform for financial analysis.

## Machine Learning Algorithms

This project uses four Machine Learning algorithms:

1. **Random Forest** – Combines multiple decision trees using majority voting.
2. **XGBoost** – Uses optimized gradient boosting to improve predictions.
3. **Support Vector Machine (SVM)** – Separates customer categories using an optimal hyperplane.
4. **Gradient Boosting** – Builds sequential models to reduce prediction errors.

## Dataset

**Dataset:** Default of Credit Card Clients

The dataset contains 30,000 customer records with information such as:

- Customer age
- Credit limit
- Education
- Marital status
- Repayment status
- Bill amount
- Previous payment amount

**Target Variable:** Default Payment Next Month

- `0` – No Default
- `1` – Default

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Streamlit

## Project Workflow

1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Selection
5. Model Training
6. Model Evaluation
7. Credit Default Prediction
8. Financial Analytics and Visualization

## Model Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Score

## Installation

Clone the repository:

```bash
git clone https://github.com/savarirokade-svg/FinSight.git
```

Navigate to the project folder:

```bash
cd FinSight
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

```bash
streamlit run app.py
```

## Project Structure

```text
FinSight/
│
├── app/
│   ├── dataset/
│   └── application files
│
├── notebooks/
├── models/
├── reports/
├── requirements.txt
├── README.md
└── .gitignore
```

*Note: Adjust the structure above to match the actual folders and filenames in your repository.*

## Future Enhancements

- Real-time credit risk prediction.
- Improved model optimization.
- Advanced financial analytics.
- Explainable AI for prediction transparency.
- Multilingual user interface.

## Project Information

**Project Name:** FinSight

**Domain:** Machine Learning | Financial Analytics

**Project Type:** Academic Machine Learning Project

**Developed By:** Savari Prafull Rokade

**GitHub:** https://github.com/savarirokade-svg/FinSight

---

⭐ If you find this project interesting, feel free to explore the repository.
