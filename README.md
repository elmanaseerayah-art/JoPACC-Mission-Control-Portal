# JoPACC Mission Control Portal

An AI-powered analytics portal for Jordanian payment systems, developed using Machine Learning, Explainable AI (XAI), Anomaly Detection, Fuzzy Logic, Reinforcement Learning, and Flask.

## Project Overview

The JoPACC Mission Control Portal analyzes payment-system data and provides intelligent prediction, validation, evaluation, and intervention capabilities.

The project works with multiple payment systems, including:

- CliQ
- JoMoPay
- eFAWATEERcom
- ACH
- ECCU

The system integrates several Artificial Intelligence techniques into a Flask-based web portal.

## Main Features

- Transaction value prediction using regression
- Payment-system activity level classification
- SHAP and LIME prediction explanations
- Anomaly-based input validation
- Fuzzy growth-quality evaluation
- Reinforcement Learning intervention recommendations
- Interactive Flask web interface

## 1. Regression

The regression component predicts:

`transaction_value_jod`

Two regression models were evaluated:

- Linear Regression
- Random Forest Regressor

The models were evaluated using:

- MAE
- RMSE
- R²

The model with the highest R² is selected while MAE and RMSE are also used to evaluate prediction error.

## 2. Classification

The classification component predicts the payment-system activity level.

The project evaluates classification approaches including:

- Tuned Random Forest
- Gradient Boosting

GridSearchCV is used to optimize the Random Forest classifier parameters.

## 3. Explainable AI – SHAP & LIME

SHAP and LIME are integrated to explain individual predictions.

They show how each input feature contributes to a prediction by identifying positive and negative feature contributions.

This makes the Machine Learning predictions easier to interpret and understand.

## 4. Anomaly Detection

Before generating a regression or classification prediction, incoming data is validated using anomaly detection.

Two techniques were evaluated:

- Isolation Forest
- One-Class SVM

Synthetic anomalies were generated for evaluation because the original dataset did not contain labeled anomalies.

The evaluation results were:

- One-Class SVM ROC-AUC: 1.0000
- Isolation Forest ROC-AUC: 0.9647

One-Class SVM was therefore selected for input validation.

If an input is classified as anomalous, the portal prevents it from continuing to the prediction process.

## 5. Fuzzy Growth Quality Evaluator

A Fuzzy Logic system was developed to evaluate the growth quality of JoMoPay.

The fuzzy model uses:

- Total Transaction Activity
- Commercial Wallet Count
- Cash-Out Transaction Count

The system evaluates growth as:

- Fragile Growth
- Unbalanced Growth
- Healthy Growth
- Sustainable Growth

Centroid defuzzification is used to generate a final Growth Quality Score from 0 to 100.

## 6. Reinforcement Learning

A tabular Q-Learning intervention simulator was developed to recommend suitable actions based on:

- Current activity level
- Payment system
- System-specific indicators
- Available budget

The state variables are divided into discrete levels such as Low, Medium, and High.

Two reward policies were evaluated:

- Improvement-Focused Policy
- Cost-Aware Policy

The Cost-Aware Policy was selected based on the project evaluation.

## Technologies

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- SHAP
- LIME
- Fuzzy Logic
- Q-Learning
- HTML
- CSS
- JavaScript
- Jupyter Notebook

## System Workflow

The general portal workflow is:

`User Input → Anomaly Validation → Prediction → SHAP/LIME Explanation → Result`

Additional modules provide:

`JoMoPay Data → Fuzzy Logic → Growth Quality Evaluation`

and

`System State + Budget → Q-Learning → Intervention Recommendation`

## Dataset

The project uses historical data for several Jordanian payment systems, including CliQ, JoMoPay, eFAWATEERcom, ACH, and ECCU.

Features represent transaction activity, users, time information, and system-specific behavior.

## Security

Sensitive credentials and API keys are not included in the public repository.

Environment variables should be used when external API credentials are required.

## Project Structure

The repository contains:

- Flask application files
- Regression and classification models
- SHAP and LIME explainability components
- Anomaly detection models
- Fuzzy Logic implementation
- Reinforcement Learning implementation
- Jupyter notebooks
- HTML templates
- Dataset
- Trained model files
- Full project report

## Author

**Ayah El-Manaseer**

Applied Analytical Models Project  
Al Hussein Technical University (HTU)
