# 🏦 Loan Approval Prediction using Machine Learning

This project applies machine learning techniques to predict whether a loan application should be approved, based on historical applicant data. The system is designed to help automate the loan approval process, reduce human error and processing delays, and identify key drivers influencing approval decisions. The project also provides valuable insights into applicant demographics, credit behaviors, and risk factors that can inform marketing and operational strategies.

---

## 🎯 Objective

- Predict loan approval status using ML models trained on historical data.
- Streamline and automate decision-making in loan processing.
- Minimize manual intervention while maintaining transparency.
- Derive business insights from feature analysis to support data-driven strategies.

---

## 📁 Dataset Overview

- **Source:** [Kaggle](https://www.kaggle.com/)
- **Size:** ~45,000 loan application records.
- **Fields:** Includes applicant details such as age, income, home ownership, education, credit history, previous defaults, loan amount, interest rate, and approval status.

---

## 🧪 Technologies & Tools

- **Programming Language:** Python
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Modeling:** Scikit-learn, XGBoost
- **Evaluation:** Confusion Matrix, ROC-AUC, F1-Score
- **Notebook:** Jupyter

---

## 🔎 Exploratory Data Analysis (EDA)

All EDA is documented in [`EDA.ipynb`](./EDA.ipynb). Key highlights include:

### ➤ Loan Status Distribution
- **Approved:** 10,000  
- **Rejected:** 35,000  
- Imbalanced data skewed toward rejections.

### ➤ Intent-Based Insights
- Most applications are for **education** and **venture** purposes.
- **Debt consolidation** and **medical** loans have the **highest approval rates**.
- **Education** and **venture** loans have the **lowest success rates**.

### ➤ Applicant Demographics
- Majority of applicants have **High School** or **Bachelor’s** degrees.
- **Gender** does not significantly affect approval probability.
- **Renters** tend to have the **highest approval rates**; homeowners have the lowest.

### ➤ Risk Factors
- **Applicants with prior loan defaults** face **100% rejection**.
- Age-related columns (e.g., age, employment experience, credit history length) show **strong multicollinearity**.

### ➤ Credit & Income
- **Credit score alone** is not a strong predictor.
- **Higher interest rates and lower income brackets** correlate with approvals — likely due to risk-adjusted interest modeling.
- Approved applicants often request slightly **smaller loan amounts**.

---

## 📈 Model Performance

Four models were tested and evaluated using precision, recall, F1-score, and accuracy.  

| Model                    | Accuracy | Precision (1) | Recall (1) | F1 Score (1) | AUC    |
|--------------------------|----------|---------------|------------|--------------|--------|
| Logistic Regression      | 88.9%    | 0.74          | 0.75       | 0.75         | -      |
| Random Forest ✅          | 92.8%    | 0.79          | 0.87       | 0.83         | 0.97   |
| Gradient Boosting        | 91.9%    | 0.78          | 0.84       | 0.81         | -      |
| XGBoost                  | 93.6%    | 0.82          | 0.88       | 0.85         | -      |

> “(1)” refers to the “Approved” class.

---

## 🧠 Final Model Choice: Random Forest ✅

Although XGBoost offered slightly higher accuracy, **Random Forest was selected** as the final model due to:

- **Interpretability**: Easier to explain decisions to stakeholders.
- **Comparable performance** with XGBoost.
- **Faster training & tuning**, making it ideal for practical deployment.

---

## 📊 Feature Importance

Top 5 features influencing predictions:

1. `previous_loan_defaults_on_file_Yes`
2. `person_income`
3. `loan_int_rate`
4. `loan_amnt`
5. `credit_score`

![Feature Importance](./5c82b470-d573-4a9e-ab2c-41cf9820a897.png)

---

## 🧪 ROC Curve

Random Forest AUC = **0.97**, indicating excellent separability between approved and rejected applications.

![ROC Curve](./121e128f-1874-4e64-81d6-d6848d9f4bc1.png)

---

## ✅ Business Impact

- **Efficiency**: Automates repetitive, manual decision tasks.
- **Speed**: Reduces time-to-decision for applicants.
- **Accuracy**: Lowers the risk of subjective human error.
- **Insights**: Surfaces important predictors for marketing and risk teams.

---

## 🚀 Future Enhancements

- Deploy as a REST API using **FastAPI** or **Flask**.
- Build a real-time dashboard with **Streamlit** for business users.
- Incorporate **SHAP** or **LIME** for explainable ML.
- Address class imbalance using **SMOTE** or **cost-sensitive learning**.
- Expand with **time-based** or **regional** segmentation to improve performance.



