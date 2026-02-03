# HR Attrition Prediction using Machine Learning (XGBoost)

![Python](https://img.shields.io/badge/Python-3.14-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-GradientBoosting-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

A predictive machine learning project that identifies employees at risk of leaving an organization using **Python, Scikit-learn, and XGBoost**. The solution generates **interpretable attrition risk scores** and **data-driven retention recommendations** to support HR decision-making.

---

## 📌 Project Overview

Employee attrition is costly and impacts productivity, hiring budgets, and team stability. This project builds a classification model to predict employee attrition and provide actionable insights for HR stakeholders.

### ✅ Highlights
- End-to-end ML pipeline: preprocessing → training → evaluation → inference
- Class imbalance handled using **SMOTE**
- Feature engineering for improved performance
- **Cross-validation** for robust evaluation
- Interpretable outputs: risk score + key drivers

---

## 🎯 Objective

- Predict whether an employee is likely to leave the organization (**Attrition: Yes/No**)
- Improve minority-class detection through oversampling
- Provide interpretable risk scores to support retention strategy

---

## 🧠 Model & Performance

**Final Model:** XGBoost Classifier  
**Validation Strategy:** Cross-validation  
**Accuracy Achieved:** **86%**

Evaluation metrics include:
- Accuracy
- Precision / Recall / F1-score
- Confusion Matrix
- ROC-AUC (optional)

---

## 🛠 Tech Stack

- **Python 3.14.3**
- **Pandas, NumPy**
- **Scikit-learn**
- **XGBoost**
- **Imbalanced-learn (SMOTE)**
- **Matplotlib / Seaborn**

---

## 📊 Dataset

This project uses the **IBM HR Analytics Employee Attrition & Performance dataset**, fetched from **Kaggle**.

- Dataset Source: IBM HR Analytics
- Platform: Kaggle
- Type: Structured HR employee dataset
- Target Variable: `Attrition` (Yes/No)

The dataset includes employee-related features such as:
- Demographics (Age, Gender, Marital Status)
- Work profile (Job Role, Department, Years at Company)
- Compensation (Monthly Income, Salary Hike)
- Engagement indicators (OverTime, Job Satisfaction, Performance Rating)

> ⚠️ Note: This dataset is publicly available on Kaggle and is used here strictly for educational/project purposes.

---

## 📂 Repository Structure

```bash
hr-attrition-prediction/
│
├── data/
│   ├── raw/                      # Raw Kaggle dataset (IBM HR Analytics)
│   └── processed/                # Cleaned dataset
│
├── notebooks/
│   ├── EDA.ipynb                 # Exploratory Data Analysis
│   └── Modeling.ipynb            # Model experiments
│
├── src/
│   ├── preprocessing.py          # Data cleaning & transformations
│   ├── feature_engineering.py    # Feature generation
│   ├── train.py                  # Model training pipeline
│   ├── evaluate.py               # Model evaluation scripts
│   └── predict.py                # Risk scoring / inference script
│
├── models/
│   └── xgb_model.pkl             # Saved trained model
│
├── reports/
│   ├── evaluation_report.md      # Results summary
│   └── feature_importance.png    # Model interpretability chart
│
├── requirements.txt
├── README.md
└── LICENSE
```
## ⚙️ Methodology

### 1) Data Preprocessing
- Missing value handling  
- Encoding categorical variables  
- Scaling numerical variables (if required)  
- Stratified train-test split  

### 2) Handling Class Imbalance
Attrition datasets are typically imbalanced.  
To improve learning for the minority class, **SMOTE (Synthetic Minority Oversampling Technique)** was applied on the training data.

### 3) Feature Engineering
Example engineered features include:
- Tenure buckets  
- Overtime + satisfaction risk indicators  
- Interaction variables for HR-relevant signals  

### 4) Model Training & Validation
- Baseline model comparison (optional)  
- Hyperparameter tuning (Grid Search / Random Search)  
- Cross-validation to reduce overfitting  

---

## 🧾 Output: Attrition Risk Score

The model generates:
- **Attrition probability score (0–1)**  
- Risk classification:
  - Low Risk  
  - Medium Risk  
  - High Risk  
- Key contributing drivers (feature importance)  
- Suggested retention actions  

---

## 🚀 How to Run

### 1) Clone Repository
```bash
git clone https://github.com/BuildWithSaravanan/hr-attrition-prediction-model.git
cd hr-attrition-prediction
```
### 2) Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```
### 3)Install Dependencies
```bash
pip install -r requirements.txt
```
### 4) Train Model
```bash
python src/train.py
```
### 5) Evaluate Model
```bash
python src/evaluate.py
```
## 📌 Business Impact

This project helps HR teams to:
- Identify employees likely to leave before attrition happens  
- Prioritize retention actions for high-risk employees  
- Reduce turnover costs  
- Improve workforce planning with data-driven insights  

---

## 🔮 Future Improvements
- Add **SHAP explainability** for per-employee interpretability  
- Deploy as a web app (Streamlit/Flask)  
- Add model monitoring (drift detection)  
- Add fairness checks across demographic groups  
- Integrate with dashboards (Power BI / Tableau)  

---

## 👤 Author

**Saravanan Srinivasan**  
📩 Email: meet.saravanan10@gmail.com  

