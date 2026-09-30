import pandas as pd
import numpy as np
import joblib
import os
 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
# Load the train/test data 
 
data = joblib.load(f"preprocessed_regression.pkl")
X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]
 
print("X_train:", X_train.shape, "| X_test:", X_test.shape)
print("y_train:", y_train.shape, "| y_test:", y_test.shape)
 
# Train Model 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
 
lr_mae = mean_absolute_error(y_test, lr_preds)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
lr_r2 = r2_score(y_test, lr_preds)
 
print("\nLinear Regression ")
print(f"MAE: {lr_mae:,.2f}")
print(f"RMSE: {lr_rmse:,.2f}")
print(f"R2: {lr_r2:.4f}")
 
# Train Model 2: Random Forest Regressor
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
 
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_r2 = r2_score(y_test, rf_preds)
 
print("\nRandom Forest Regressor ")
print(f"MAE: {rf_mae:,.2f}")
print(f"RMSE: {rf_rmse:,.2f}")
print(f"R2: {rf_r2:.4f}")
 

# Compare and select the best model
results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [lr_mae, rf_mae],
    "RMSE": [lr_rmse, rf_rmse],
    "R2": [lr_r2, rf_r2]
})
print("\n--- Model comparison ---")
print(results.to_string(index=False))
 
best_row = results.loc[results["R2"].idxmax()]
best_model_name = best_row["Model"]
best_model = lr_model if best_model_name == "Linear Regression" else rf_model
print(f"\nBest model selected: {best_model_name} (R2 = {best_row['R2']:.4f})")
 
 
# Save the best model
os.makedirs("models", exist_ok=True)
joblib.dump(
    best_model,
    "models/best_regression_model.pkl"
)

print("\nSaved best regression model to: "
    "models/best_regression_model.pkl"
)
