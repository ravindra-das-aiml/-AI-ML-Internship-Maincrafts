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

## 📁 Task 3 — Model Validation, Overfitting Control & Hyperparameter Tuning

**Goal:** Go deeper into professional ML practice — detect and control overfitting, validate models
reliably using cross-validation, and tune hyperparameters systematically with GridSearchCV.

**What was done:**
- Trained an unconstrained Decision Tree to demonstrate overfitting (train RMSE ≈ $0, test RMSE ≈ $69,790).
- Applied 5-fold cross-validation for a reliable, split-independent performance estimate.
- Used `GridSearchCV` to tune `max_depth` and `min_samples_split`, finding the best combination.
- Compared the tuned model against the Task-2 baselines (Linear Regression, Ridge Regression) and the untuned tree.
- Justified the final model choice using RMSE, R², overfitting gap, and cross-validation reliability.

**Model Comparison:**

| Model | Train RMSE | Test RMSE | Test R² | Overfit Gap |
|---|---|---|---|---|
| Linear Regression | 68,435 | 70,061 | 0.6254 | 1,627 |
| Ridge Regression | 68,434 | 70,057 | 0.6255 | 1,623 |
| Decision Tree (untuned) | ~0 | 69,790 | 0.629 | 69,790 |
| **Decision Tree (tuned)** ✅ (best) | 49,498 | 60,997 | 0.7161 | 11,499 |

**Files:**
- [`AI_ML_Task3_Model_Validation_Tuning.ipynb`](./task3/AI_ML_Task3_Model_Validation_Tuning.ipynb) — full notebook
- [`Task3_ML_Report.pdf`](./task3/Task3_ML_Report.pdf) — 2-3 page report
- [`best_tuned_model.joblib`](./task3/best_tuned_model.joblib), [`scaler.joblib`](./task3/scaler.joblib) — saved tuned model + scaler

---

## 📁 Task 4 — Classification Models, Evaluation Metrics & Handling Imbalanced Data

**Goal:** Move from regression to classification — build a binary classification system and evaluate
it properly with metrics beyond accuracy, handle class imbalance, and compare classifiers.

**What was done:**
- Built a binary classifier on the Breast Cancer Wisconsin dataset (Malignant vs Benign).
- Evaluated with confusion matrix, precision, recall, F1-score, and ROC-AUC.
- Explained why accuracy alone is misleading on imbalanced data.
- Tested class weighting (`class_weight="balanced"`) as an imbalance-handling technique.
- Compared Logistic Regression vs Decision Tree Classifier and justified the final model choice.

**Model Comparison:**

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Logistic Regression (baseline)** ✅ (best) | 0.9825 | 0.9861 | 0.9861 | 0.9861 |
| Logistic Regression (balanced) | 0.9561 | 0.9855 | 0.9444 | 0.9645 |
| Decision Tree | 0.9123 | 0.9559 | 0.9028 | 0.9286 |

ROC-AUC (baseline Logistic Regression): **0.995**

**Files:**
- [`AI_ML_Task4_Classification_Evaluation.ipynb`](./task4/AI_ML_Task4_Classification_Evaluation.ipynb) — full notebook
- [`Task4_ML_Report.pdf`](./task4/Task4_ML_Report.pdf) — 2-page report
- [`logistic_model.joblib`](./task4/logistic_model.joblib), [`logistic_balanced_model.joblib`](./task4/logistic_balanced_model.joblib), [`scaler.joblib`](./task4/scaler.joblib) — saved models

---

## 📁 Task 5 — Ensemble Learning, Model Optimization & Real-World ML Pipeline

**Goal:** Build a complete classification system with ensemble models and hyperparameter optimization, then select the final model with evidence.

**What was done:**
- Trained Logistic Regression (baseline), Random Forest and Gradient Boosting on the Breast Cancer dataset.
- Tuned Random Forest with `GridSearchCV` (best: `max_depth=5, n_estimators=100`).
- Compared models on accuracy, precision, recall, F1, ROC-AUC and 5-fold cross-validation (using the sklearn `Pipeline` API).
- Added feature importance visualization for model interpretation.
- Selected the final model on performance, stability and interpretability.

**Model Comparison:**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV F1 (5-fold) |
|---|---|---|---|---|---|---|
| **Logistic Regression** ✅ (final) | 0.9825 | 0.9861 | 0.9861 | 0.9861 | 0.9954 | 0.985 |
| Random Forest | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9937 | 0.965 |
| Gradient Boosting | 0.9561 | 0.9467 | 0.9861 | 0.9660 | 0.9907 | 0.971 |
| Random Forest (tuned) | 0.9561 | 0.9589 | 0.9722 | 0.9655 | 0.9934 | 0.968 |

Key insight: on this small, nearly linearly separable dataset, the simple baseline beat the ensembles — ensembles are benchmarked, not assumed better.

**Files:**
- [`AI_ML_Task5_Ensemble_Learning.ipynb`](./Task5/AI_ML_Task5_Ensemble_Learning.ipynb) — full notebook
- [`Task5_ML_Report.pdf`](./Task5/Task5_ML_Report.pdf) — 3-page report
- `final_logistic_model.joblib`, `best_rf_model.joblib`, `scaler.joblib` — saved models

---

## 🛠️ Tools & Technologies

Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter Notebook

## 📊 Dataset

[California Housing Dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)
— predicting median house value from features like median income, house age, rooms, population, and location.
