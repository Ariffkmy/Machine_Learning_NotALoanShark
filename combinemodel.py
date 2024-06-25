import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


df = pd.read_csv('encoded_dataset.csv')

# Data separatio
y = df['repay_fail']
X = df.drop('repay_fail', axis=1)

# Splitting dataset into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)

y_train_pred_dt = dt_regressor.predict(X_train)
y_test_pred_dt = dt_regressor.predict(X_test)

train_mse_dt = mean_squared_error(y_train, y_train_pred_dt)
test_mse_dt = mean_squared_error(y_test, y_test_pred_dt)
train_r2_dt = r2_score(y_train, y_train_pred_dt)
test_r2_dt = r2_score(y_test, y_test_pred_dt)

# Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)

y_train_pred_rf = rf_regressor.predict(X_train)
y_test_pred_rf = rf_regressor.predict(X_test)

train_mse_rf = mean_squared_error(y_train, y_train_pred_rf)
test_mse_rf = mean_squared_error(y_test, y_test_pred_rf)
train_r2_rf = r2_score(y_train, y_train_pred_rf)
test_r2_rf = r2_score(y_test, y_test_pred_rf)

# Linear Regresssion
lr = LinearRegression()
lr.fit(X_train, y_train)

y_train_pred_lr = lr.predict(X_train)
y_test_pred_lr = lr.predict(X_test)

train_mse_lr = mean_squared_error(y_train, y_train_pred_lr)
test_mse_lr = mean_squared_error(y_test, y_test_pred_lr)
train_r2_lr = r2_score(y_train, y_train_pred_lr)
test_r2_lr = r2_score(y_test, y_test_pred_lr)

# Comparison of Model Performanc
print("Decision Tree Regression Performance:")
print("Training set Mean Squared Error (MSE):", train_mse_dt)
print("Test set Mean Squared Error (MSE):", test_mse_dt)
print("Training set R-squared (R2):", train_r2_dt)
print("Test set R-squared (R2):", test_r2_dt)

print("\nRandom Forest Regression Performance:")
print("Training set Mean Squared Error (MSE):", train_mse_rf)
print("Test set Mean Squared Error (MSE):", test_mse_rf)
print("Training set R-squared (R2):", train_r2_rf)
print("Test set R-squared (R2):", test_r2_rf)

print("\nLinear Regression Performance:")
print("Training set Mean Squared Error (MSE):", train_mse_lr)
print("Test set Mean Squared Error (MSE):", test_mse_lr)
print("Training set R-squared (R2):", train_r2_lr)
print("Test set R-squared (R2):", test_r2_lr)

new_data = pd.read_csv('new_data.csv')


# Make predictions with the trained models
new_data_predictions_dt = dt_regressor.predict(new_data)
new_data_predictions_rf = rf_regressor.predict(new_data)
new_data_predictions_lr = lr.predict(new_data)

new_data['Predicted_repay_fail_DT'] = new_data_predictions_dt
new_data['Predicted_repay_fail_RF'] = new_data_predictions_rf
new_data['Predicted_repay_fail_LR'] = new_data_predictions_lr

# Display the new data with predictions
print(new_data.head())


# Cross-validation for Randm Forest Regressor
cv_scores_rf = cross_val_score(rf_regressor, X, y, cv=5, scoring='r2')
print("\nRandom Forest Cross-Validation R2 Scores:", cv_scores_rf)
print("Average Cross-Validation R2 Score:", cv_scores_rf.mean())

# Cross-validation for Decision Tree Regressor
cv_scores_dt = cross_val_score(dt_regressor, X, y, cv=5, scoring='r2')
print("\nDecision Tree Cross-Validation R2 Scores:", cv_scores_dt)
print("Average Cross-Validation R2 Score:", cv_scores_dt.mean())

# Cros-validation for Linear Regression
cv_scores_lr = cross_val_score(lr, X, y, cv=5, scoring='r2')
print("\nLinear Regression Cross-Validation R2 Scores:", cv_scores_lr)
print("Average Cross-Validation R2 Score:", cv_scores_lr.mean())

