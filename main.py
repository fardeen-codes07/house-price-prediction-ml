import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge
data = pd.read_csv(r"C:\Users\FARDEEN\OneDrive\Desktop\github\house-price-prediction\House Price Prediction Dataset.csv")
print(data.head())
data = data.drop(columns=["Id"])
data = pd.get_dummies(data, columns=["Location", "Condition", "Garage"], drop_first=True)
X = data.drop(columns=["Price"])
y = data["Price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print("Model Evaluation Results")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")
coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)
print("\nFeature Importance (Linear Regression)")
print(coeff_df)
train_r2 = model.score(X_train, y_train)
test_r2  = model.score(X_test, y_test)
print("\nTrain vs Test R² Scores")
print(f"Train R²: {train_r2:.4f}")
print(f"Test  R²: {test_r2:.4f}")
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
ridge_mae  = mean_absolute_error(y_test, y_pred_ridge)
ridge_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
ridge_r2   = r2_score(y_test, y_pred_ridge)

print("\nRidge Regression Results")
print(f"MAE  : {ridge_mae:.2f}")
print(f"RMSE : {ridge_rmse:.2f}")
print(f"R²   : {ridge_r2:.4f}")
print("\nModel Comparison Summary")

comparison_df = pd.DataFrame({
    "Model": ["Linear Regression", "Ridge Regression"],
    "MAE": [mae, ridge_mae],
    "RMSE": [rmse, ridge_rmse],
    "R²": [r2, ridge_r2]
})
print(comparison_df)
if ridge_r2 > r2:
    print("\nFinal Model Selected: Ridge Regression")
    print("Reason: Better generalization due to regularization.")
else:
    print("\nFinal Model Selected: Linear Regression")
    print("Reason: Comparable performance with simpler interpretation.")
#example:
sample_input = X_test.iloc[[0]]
predicted_price = ridge_model.predict(sample_input)[0]
actual_price = y_test.iloc[0]

print(f"Predicted Price: ₹ {predicted_price:,.2f}")
print(f"Actual Price: ₹ {actual_price:,.2f}")



