
from flask import Flask, render_template, request, jsonify
import os
import joblib
import pandas as pd
import plotly.express as px

app = Flask(__name__)

# ==========================================
# 1. PROJECT PATHS AND MODEL LOADING
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "gradient_boosting.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR, "dataset", "credit_default_clean.csv"
)

FEATURE_PATH = os.path.join(
    BASE_DIR, "dataset", "X_train.csv"
)

# Load trained Gradient Boosting model
model = joblib.load(MODEL_PATH)

# Get the exact feature order used during training
FEATURE_COLUMNS = pd.read_csv(
    FEATURE_PATH, nrows=0
).columns.tolist()


# ==========================================
# 2. HOME PAGE
# ==========================================

@app.route("/")
def home():
    return render_template("index.html")


# ==========================================
# 3. HEALTH CHECK
# ==========================================

@app.route("/health")
def health():
    return jsonify({
        "status": "FinSight is running",
        "model": "Gradient Boosting",
        "model_loaded": model is not None
    })


# ==========================================
# 4. CREDIT DEFAULT PREDICTION
# ==========================================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    # Display prediction form
    if request.method == "GET":
        return render_template("predict.html")

    try:
        # Read numerical customer information
        input_data = {
            "LIMIT_BAL": float(request.form["LIMIT_BAL"]),
            "AGE": int(request.form["AGE"])
        }

        # Payment status history
        payment_status_columns = [
            "PAY_0", "PAY_2", "PAY_3",
            "PAY_4", "PAY_5", "PAY_6"
        ]

        for col in payment_status_columns:
            input_data[col] = int(request.form[col])

        # Bill amounts
        for i in range(1, 7):
            col = f"BILL_AMT{i}"
            input_data[col] = float(request.form[col])

        # Previous payment amounts
        for i in range(1, 7):
            col = f"PAY_AMT{i}"
            input_data[col] = float(request.form[col])

        # Read categorical information
        sex = int(request.form["SEX"])
        education = int(request.form["EDUCATION"])
        marriage = int(request.form["MARRIAGE"])

        # Validate categorical inputs
        if sex not in [1, 2]:
            raise ValueError("Invalid gender")

        if education not in [1, 2, 3, 4]:
            raise ValueError("Invalid education")

        if marriage not in [1, 2, 3]:
            raise ValueError("Invalid marital status")

        # One-hot encode gender
        for value in [1, 2]:
            input_data[f"SEX_{value}"] = int(sex == value)

        # One-hot encode education
        for value in [1, 2, 3, 4]:
            input_data[f"EDUCATION_{value}"] = int(
                education == value
            )

        # One-hot encode marital status
        for value in [1, 2, 3]:
            input_data[f"MARRIAGE_{value}"] = int(
                marriage == value
            )

        # Arrange features in exactly the training order
        input_df = pd.DataFrame(
            [input_data],
            columns=FEATURE_COLUMNS
        )

        # Check for missing features
        if input_df.isnull().any().any():
            raise ValueError("Some customer inputs are missing")

        # Predict using Gradient Boosting
        prediction = int(model.predict(input_df)[0])

        probability = (
            float(model.predict_proba(input_df)[0][1]) * 100
        )

        # Convert prediction into readable result
        if prediction == 1:
            result = "Higher Default Risk"
        else:
            result = "Lower Default Risk"

        return render_template(
            "predict.html",
            result=result,
            prediction=prediction,
            probability=round(probability, 2)
        )

    except (ValueError, KeyError) as error:

        return render_template(
            "predict.html",
            error=f"Please check all input fields. Details: {error}"
        )


# ==========================================
# 5. ANALYTICS DASHBOARD
# ==========================================

@app.route("/analytics")
def analytics():

    df = pd.read_csv(DATA_PATH)

    target = "default payment next month"

    total = len(df)
    defaults = int((df[target] == 1).sum())
    non_defaults = int((df[target] == 0).sum())

    default_rate = round(
        (defaults / total) * 100, 2
    )

    # Chart 1: Customer default distribution
    distribution = pd.DataFrame({
        "Customer Status": ["No Default", "Default"],
        "Customers": [non_defaults, defaults]
    })

    fig1 = px.pie(
        distribution,
        names="Customer Status",
        values="Customers",
        hole=0.55,
        title="Customer Default Distribution"
    )

    # Chart 2: Customer age distribution
    fig2 = px.histogram(
        df,
        x="AGE",
        color=target,
        nbins=25,
        title="Customer Age Distribution",
        labels={target: "Default Status"}
    )

    # Chart 3: Average credit limit
    credit = (
        df.groupby(target)["LIMIT_BAL"]
        .mean()
        .reset_index()
    )

    credit["Status"] = credit[target].map({
        0: "No Default",
        1: "Default"
    })

    fig3 = px.bar(
        credit,
        x="Status",
        y="LIMIT_BAL",
        color="Status",
        title="Average Credit Limit by Default Status",
        labels={
            "LIMIT_BAL": "Average Credit Limit"
        }
    )

    # Chart 4: Monthly payment trends
    payment_columns = [
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
        "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]

    payment_data = pd.DataFrame({
        "Month": [
            "Month 1", "Month 2", "Month 3",
            "Month 4", "Month 5", "Month 6"
        ],
        "Average Payment": [
            df[col].mean() for col in payment_columns
        ]
    })

    fig4 = px.line(
        payment_data,
        x="Month",
        y="Average Payment",
        markers=True,
        title="Average Customer Payments"
    )

    # Convert charts into HTML
    charts = [
        fig1.to_html(
            full_html=False,
            include_plotlyjs="cdn"
        ),

        fig2.to_html(
            full_html=False,
            include_plotlyjs=False
        ),

        fig3.to_html(
            full_html=False,
            include_plotlyjs=False
        ),

        fig4.to_html(
            full_html=False,
            include_plotlyjs=False
        )
    ]

    return render_template(
        "analytics.html",
        total=total,
        defaults=defaults,
        non_defaults=non_defaults,
        default_rate=default_rate,
        charts=charts
    )


# ==========================================
# 6. MODEL INTELLIGENCE
# ==========================================

@app.route("/models")
def models():

    # Read actual model evaluation results
    REPORT_PATH = os.path.join(
        BASE_DIR, "reports", "day5_evaluation_report.csv"
    )

    model_df = pd.read_csv(REPORT_PATH)

    # Find the best model based on F1 Score
    best_index = model_df["F1 Score"].idxmax()
    best_model = model_df.loc[best_index, "Model"]
    best_f1 = round(
        float(model_df.loc[best_index, "F1 Score"]), 4
    )

    # Prepare data for metric comparison chart
    metric_columns = [
        "Accuracy", "Precision", "Recall",
        "F1 Score", "ROC-AUC", "PR-AUC"
    ]

    comparison_data = model_df.melt(
        id_vars="Model",
        value_vars=metric_columns,
        var_name="Metric",
        value_name="Score"
    )

    fig1 = px.bar(
        comparison_data,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title="ML Model Performance Comparison",
        range_y=[0, 1]
    )

    # Prepare ROC-AUC vs PR-AUC chart
    fig2 = px.scatter(
        model_df,
        x="ROC-AUC",
        y="PR-AUC",
        color="Model",
        text="Model",
        title="ROC-AUC vs PR-AUC Comparison",
        range_x=[0, 1],
        range_y=[0, 1]
    )

    fig2.update_traces(
        textposition="top center",
        marker=dict(size=13)
    )

    # Convert charts into HTML
    charts = [
        fig1.to_html(
            full_html=False,
            include_plotlyjs="cdn"
        ),
        fig2.to_html(
            full_html=False,
            include_plotlyjs=False
        )
    ]

    # Send results to the HTML page
    return render_template(
        "models.html",
        models=model_df.to_dict(orient="records"),
        best_model=best_model,
        best_f1=best_f1,
        charts=charts
    )


# ==========================================
# 7. RUN FLASK APPLICATION
# ==========================================

if __name__ == "__main__":
    app.run(debug=True)
