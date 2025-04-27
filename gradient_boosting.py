import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

# Load the CSV file
data = pd.read_csv('KO_1919-09-06_2025-03-15.csv')

# Ensure the 'date' column is in datetime format with UTC
data['date'] = pd.to_datetime(data['date'], utc=True)

# Add derived features
data['log_volume'] = np.log1p(data['volume'])
data['lag_close_1'] = data['close'].shift(1)  # Previous day's close
data['volatility_5'] = data['close'].rolling(window=5).std()  # 5-day volatility
data['returns'] = (data['close'] - data['open']) / data['open']  # Daily returns
data['lag_return_5'] = data['returns'].shift(5)  # 5-day lagged return for momentum

# Remove outliers: Drop extreme returns (beyond 3 standard deviations)
return_std = data['returns'].std()
data = data[abs(data['returns']) <= 3 * return_std]

# Drop rows with NaN values
data = data.dropna()

# Filter the data for training (2000-2007) and testing (2008-2010)
train_data = data[(data['date'] >= '2000-01-01') & (data['date'] <= '2002-12-31')]
test_data = data[(data['date'] >= '2003-01-01') & (data['date'] <= '2003-07-31')]

# Define features (X) and target (y)
features = ['lag_close_1', 'log_volume', 'volatility_5', 'lag_return_5']  # Simplified features
X_train = train_data[features]
y_train = train_data['returns']
X_test = test_data[features + ['open']]  # Include 'open' for price calculation
y_test = test_data['returns']
y_test_close = test_data['close']  # Actual close prices

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test[features])

# Define Gradient Boosting Regressor
base_model = GradientBoostingRegressor(
    learning_rate=0.01,
    max_depth=2,
    min_samples_split=10,
    min_samples_leaf=20,  # Increased to prevent overfitting
    subsample=0.7,
    validation_fraction=0.2,
    n_iter_no_change=5,
    random_state=42
)

# Perform grid search with time-series cross-validation
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [1, 2],
    'learning_rate': [0.005, 0.01],
    'subsample': [0.6, 0.7]
}
tscv = TimeSeriesSplit(n_splits=5)
grid_search = GridSearchCV(
    base_model, param_grid, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

# Best model
model = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)

# Cross-validation
cv_scores = []
for train_idx, val_idx in tscv.split(X_train_scaled):
    X_cv_train, X_cv_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    model.fit(X_cv_train, y_cv_train)
    y_cv_pred = model.predict(X_cv_val)
    cv_mse = mean_squared_error(y_cv_val, y_cv_pred)
    cv_scores.append(cv_mse)
print(f"Cross-Validation MSE Scores: {cv_scores}")
print(f"Average CV MSE: {np.mean(cv_scores):.6f}")

# Train the model
model.fit(X_train_scaled, y_train)

# Predict returns
y_pred = model.predict(X_test_scaled)

# Convert to predicted close prices
y_pred_close = X_test['open'] * (1 + y_pred)

# Calculate metrics for returns
train_pred = model.predict(X_train_scaled)
train_mse = mean_squared_error(y_train, train_pred)
test_mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Train Mean Squared Error (Returns): {train_mse:.6f}")
print(f"Test Mean Squared Error (Returns): {test_mse:.6f}")
print(f"Test R² Score (Returns): {r2:.4f}")

# Calculate MSE for close prices
mse_close = mean_squared_error(y_test_close, y_pred_close)
print(f"Test Mean Squared Error (Close Prices): {mse_close:.6f}")

# Baseline: Predict previous day's close
baseline_pred_close = test_data['lag_close_1']
baseline_mse_close = mean_squared_error(y_test_close, baseline_pred_close)
print(f"Baseline MSE (Close Prices, Previous Day): {baseline_mse_close:.6f}")

# Calculate directional accuracy
directional_accuracy = np.mean(np.sign(y_test) == np.sign(y_pred)) * 100
print(f"Directional Accuracy: {directional_accuracy:.2f}%")

# Plot actual vs. predicted close prices
plt.figure(figsize=(12, 6))
plt.plot(test_data['date'], y_test_close, label='Actual Close Prices', color='blue', alpha=0.6)
plt.plot(test_data['date'], y_pred_close, label='Predicted Close Prices', color='red', alpha=0.6)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Gradient Boosting Actual vs Predicted Close Prices')
plt.legend()
plt.grid()
plt.show()

# Feature importance
feature_importance = model.feature_importances_
print("Feature Importance:")
for name, importance in zip(features, feature_importance):
    print(f"{name}: {importance:.4f}")

# Feature correlation matrix
print("\nFeature Correlation Matrix:")
print(train_data[features].corr())