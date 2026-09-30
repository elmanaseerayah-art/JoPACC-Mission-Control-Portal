import pandas as pd
import joblib
import os
import json

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# 1. Load the preprocessed data

data = joblib.load(
    "preprocessed_classification.pkl"
)

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]


# Load decoding dictionary

with open(
    "decoding_dicts.json",
    "r"
) as file:
    decoding = json.load(file)


activity_decoding = {
    int(key): value
    for key, value
    in decoding["activity_decoding"].items()
}

class_labels = sorted(
    activity_decoding.keys()
)

class_names = [
    activity_decoding[label]
    for label in class_labels
]


print(
    "X_train:",
    X_train.shape,
    "| X_test:",
    X_test.shape
)

print(
    "y_train:",
    y_train.shape,
    "| y_test:",
    y_test.shape
)

print("\nClass distribution in training data:")

print(
    y_train.value_counts().sort_index()
)

print("\nClass distribution in testing data:")

print(
    y_test.value_counts().sort_index()
)


# 2. Model 1: Logistic Regression

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


logreg_model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

logreg_model.fit(
    X_train_scaled,
    y_train
)

logreg_preds = logreg_model.predict(
    X_test_scaled
)


logreg_acc = accuracy_score(
    y_test,
    logreg_preds
)

logreg_prec = precision_score(
    y_test,
    logreg_preds,
    average="macro",
    zero_division=0
)

logreg_rec = recall_score(
    y_test,
    logreg_preds,
    average="macro",
    zero_division=0
)

logreg_f1 = f1_score(
    y_test,
    logreg_preds,
    average="macro",
    zero_division=0
)


print("\n--- Logistic Regression ---")

print(f"Accuracy:  {logreg_acc:.4f}")
print(f"Precision: {logreg_prec:.4f}")
print(f"Recall:    {logreg_rec:.4f}")
print(f"F1-score:  {logreg_f1:.4f}")

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        logreg_preds,
        labels=class_labels
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        logreg_preds,
        labels=class_labels,
        target_names=class_names,
        zero_division=0
    )
)


# 3. Model 2: Random Forest with Hyperparameter Tuning

rf_model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)


param_grid = {
    "n_estimators": [
        100,
        200,
        300,
        500
    ],

    "max_depth": [
        None,
        5,
        8,
        12
    ],

    "min_samples_split": [
        2,
        5,
        10
    ],

    "min_samples_leaf": [
        1,
        2,
        4
    ],

    "class_weight": [
        None,
        "balanced"
    ]
}


cv_strategy = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv_strategy,
    n_jobs=-1,
    verbose=1
)


grid_search.fit(
    X_train,
    y_train
)


best_rf_model = grid_search.best_estimator_


print("\nBest Random Forest parameters:")

print(
    grid_search.best_params_
)

print(
    "Best cross-validation F1-score:",
    round(grid_search.best_score_, 4)
)


rf_preds = best_rf_model.predict(
    X_test
)


rf_acc = accuracy_score(
    y_test,
    rf_preds
)

rf_prec = precision_score(
    y_test,
    rf_preds,
    average="macro",
    zero_division=0
)

rf_rec = recall_score(
    y_test,
    rf_preds,
    average="macro",
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    rf_preds,
    average="macro",
    zero_division=0
)


print("\n--- Tuned Random Forest Classifier ---")

print(f"Accuracy:  {rf_acc:.4f}")
print(f"Precision: {rf_prec:.4f}")
print(f"Recall:    {rf_rec:.4f}")
print(f"F1-score:  {rf_f1:.4f}")

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        rf_preds,
        labels=class_labels
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        rf_preds,
        labels=class_labels,
        target_names=class_names,
        zero_division=0
    )
)


# 4. Model 3: Gradient Boosting Classifier

gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)


gb_model.fit(
    X_train,
    y_train
)


gb_preds = gb_model.predict(
    X_test
)


gb_acc = accuracy_score(
    y_test,
    gb_preds
)

gb_prec = precision_score(
    y_test,
    gb_preds,
    average="macro",
    zero_division=0
)

gb_rec = recall_score(
    y_test,
    gb_preds,
    average="macro",
    zero_division=0
)

gb_f1 = f1_score(
    y_test,
    gb_preds,
    average="macro",
    zero_division=0
)


print("\n--- Gradient Boosting Classifier ---")

print(f"Accuracy:  {gb_acc:.4f}")
print(f"Precision: {gb_prec:.4f}")
print(f"Recall:    {gb_rec:.4f}")
print(f"F1-score:  {gb_f1:.4f}")

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        gb_preds,
        labels=class_labels
    )
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        gb_preds,
        labels=class_labels,
        target_names=class_names,
        zero_division=0
    )
)


# 5. Compare all models

results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Tuned Random Forest",
        "Gradient Boosting"
    ],

    "Accuracy": [
        logreg_acc,
        rf_acc,
        gb_acc
    ],

    "Precision": [
        logreg_prec,
        rf_prec,
        gb_prec
    ],

    "Recall": [
        logreg_rec,
        rf_rec,
        gb_rec
    ],

    "F1-score": [
        logreg_f1,
        rf_f1,
        gb_f1
    ]
})


print("\n--- Model Comparison ---")

print(
    results.to_string(
        index=False
    )
)


# 6. Select the best model

best_row = results.loc[
    results["F1-score"].idxmax()
]

best_model_name = best_row["Model"]


if best_model_name == "Logistic Regression":

    best_model = logreg_model

elif best_model_name == "Tuned Random Forest":

    best_model = best_rf_model

else:

    best_model = gb_model


print(
    f"\nBest model selected: {best_model_name}"
)

print(
    f"Accuracy: {best_row['Accuracy']:.4f}"
)

print(
    f"Precision: {best_row['Precision']:.4f}"
)

print(
    f"Recall: {best_row['Recall']:.4f}"
)

print(
    f"F1-score: {best_row['F1-score']:.4f}"
)


# 7. Save the best model

os.makedirs(
    "models",
    exist_ok=True
)


joblib.dump(
    best_model,
    "models/best_classification_model.pkl"
)


print(
    "\nSaved best classification model to: "
    "models/best_classification_model.pkl"
)


# Logistic Regression requires the fitted scaler

if best_model_name == "Logistic Regression":

    joblib.dump(
        scaler,
        "models/classification_scaler.pkl"
    )

    print(
        "Saved classification scaler to: "
        "models/classification_scaler.pkl"
    )

 
