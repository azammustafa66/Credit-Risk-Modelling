# 💳 Credit Risk Model

This project is an **end-to-end machine learning solution** that predicts the **probability of loan default** based on borrower profiles and loan characteristics. It automates the process of cleaning financial data, engineering meaningful risk features, training a classification model, and preparing the pipeline for deployment or integration.

---

## 🎯 Project Overview

* **Goal:** Build a predictive model to assess a customer’s likelihood of default.
* **Approach:** Perform thorough data cleaning, handle missing/outlier values, engineer financial ratios, reduce multicollinearity, and train a robust classification model.
* **Deployment:** Designed to integrate seamlessly into an app or API for real-time credit risk scoring.

---

## 🧠 Key Steps

1. **Load and Inspect Data**
   * Load loan and applicant data for model training.
   * Identify the target variable (loan default / non-default).
2. **Data Preprocessing**
   * Handle missing and duplicate records.
   * Remove irrelevant or ID columns with no predictive influence.
   * Correct errors in categorical columns (like  *loan purpose* ).
3. **Outlier Detection & Treatment**
   * Used **box plots** and **IQR-based filtering** to remove extreme outliers, especially in *processing fee* and  *income* .
4. **Feature Engineering**
   * Created financial health indicators:
     * `Loan-to-Income (LTI)` ratio
     * `Delinquency Ratio`
     * `Average Days Past Due (DPD)` per delinquency
   * Processed age column and standardized numeric features.
5. **Multicollinearity Check (VIF)**
   * Computed **Variance Inflation Factor (VIF)** to identify and remove correlated features, improving model interpretability and stability.
6. **Model Training and Evaluation**
   * Trained classification models (e.g., Logistic Regression, Decision Tree, or XGBoost).
   * Evaluated model using accuracy, recall, precision, and ROC-AUC metrics.

---

## 🧩 Project Structure

```
project/
│
├── artifacts/                       # Trained model and scaler files
│   ├── model_credit_risk.joblib
│   └── scaler_credit_risk.joblib
│
├── dataset/                         # Input data
│   └── credit_risk_data.csv
│
├── credit_risk_model_codebasics.ipynb   # Jupyter notebook (main project)
├── app.py                            # (optional) Streamlit or Flask UI for deployment
├── helper.py                         # Preprocessing and prediction logic
├── requirements.txt                  # Dependencies
└── README.md                         # Project documentation
```

---

## ⚙️ Setup Instructions

1. **Clone this repository**
   ```bash
   git clone <repo_url>
   cd project
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the notebook**
   ```bash
   jupyter notebook credit_risk_model_codebasics.ipynb
   ```
4. **(Optional) Run the app**
   ```bash
   streamlit run app.py
   ```

---

## 🧮 Example Use Case

A financial institution can integrate this model into their loan application system to instantly evaluate applicant risk levels.

Input applicant data → process through the model → output a credit risk score or “approve/deny” recommendation.

---

## 📊 Key Insights

* High **Loan-to-Income ratio** and **high Delinquency ratio** are strong predictors of default.
* Removing multicollinear variables improved model performance and interpretability.
* Data cleaning (especially outlier and missing-value handling) significantly improved stability.

---

## 🧰 Tech Stack

**Languages & Tools:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib

**ML Techniques:** Classification, Feature Engineering, Outlier Handling, Multicollinearity Reduction, VIF Analysis
