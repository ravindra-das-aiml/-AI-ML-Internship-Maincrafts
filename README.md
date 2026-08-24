# AI/ML Internship — Maincrafts Technology

**Intern:** Ravindra Das
**Domain:** Artificial Intelligence & Machine Learning
**Company:** Maincrafts Technology

This repository contains my completed tasks for the AI/ML Internship program, focused on building
and evaluating machine learning models on the **California Housing dataset**.

---

## 📁 Task 1 — Linear Regression Model (House Price Predictor)

**Goal:** Introduce the end-to-end ML workflow — data loading, EDA, preprocessing, training,
evaluation, and reporting.

**What was done:**
- Loaded the California Housing dataset and performed exploratory data analysis (distributions,
  correlations, missing values).
- Handled missing values and encoded categorical features.
- Trained a `LinearRegression` model with an 80/20 train-test split.
- Evaluated the model using MAE, RMSE, and R².
- Visualized results with Actual vs Predicted and residual plots.

**Results:** MAE ≈ $50,671 · RMSE ≈ $70,061 · R² ≈ 0.625

**Files:**
- [`task1_ml_linear_regression.ipynb`](./task1/task1_ml_linear_regression.ipynb) — full notebook
- [`Task1_ML_Report.pdf`](./task1/Task1_ML_Report.pdf) — 2-4 page report
- [`model.pkl`](./task1/model.pkl) — saved trained model

---

## 📁 Task 2 — Feature Engineering, Model Optimization & Performance Comparison

**Goal:** Go beyond a single model — apply feature scaling, train multiple algorithms, and select
the best-performing one using measurable metrics.

**What was done:**
- Applied `StandardScaler` for feature scaling.
- Trained and compared three models: Linear Regression, Ridge Regression, and Decision Tree Regressor.
- Evaluated all models on the same test set using MAE, RMSE, and R².
- Selected the best-performing model with justification.

**Model Comparison:**

| Model | MAE ($) | RMSE ($) | R² Score |
|---|---|---|---|
| **Ridge Regression** ✅ (best) | 50,668 | 70,057 | 0.6255 |
| Linear Regression | 50,671 | 70,061 | 0.6254 |
| Decision Tree (depth=5) | 50,341 | 71,511 | 0.6098 |

**Files:**
- [`AI_ML_Task2_Model_Comparison.ipynb`](./task2/AI_ML_Task2_Model_Comparison.ipynb) — full notebook
- [`Task2_ML_Report.pdf`](./task2/Task2_ML_Report.pdf) — 1-2 page report
- [`best_model.joblib`](./task2/best_model.joblib), [`scaler.joblib`](./task2/scaler.joblib) — saved best model + scaler

---

## 🛠️ Tools & Technologies

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter Notebook

## 📊 Dataset

[California Housing Dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
— predicting median house value from features like median income, house age, rooms, population, and location.
