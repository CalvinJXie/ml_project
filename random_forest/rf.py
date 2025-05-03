import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc


# Load the CSV file
data = pd.read_csv('KO_1919-09-06_2025-03-15.csv')

# Ensure the 'date' column is in datetime format with UTC
data['date'] = pd.to_datetime(data['date'], utc=True)

# Add derived features
data['log_volume'] = np.log1p(data['volume'])
data['lag_close_1'] = data['close'].shift(1)
data['volatility_5'] = data['close'].rolling(window=5).std()

# Target: Price difference (close_t - close_t-1)
data['price_diff'] = data['close'] - data['lag_close_1']

# Technical indicators
# 14-day RSI
def calculate_rsi(data, periods=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
data['rsi_14'] = calculate_rsi(data)

# 20-day moving average deviation
data['ma_20'] = data['close'].rolling(window=20).mean()
data['ma_deviation'] = data['close'] - data['ma_20']

# Date-derived features
data['day_of_week'] = data['date'].dt.dayofweek
data['month'] = data['date'].dt.month
data['year'] = data['date'].dt.year

# Remove outliers: Drop extreme price differences (beyond 3 standard deviations)
price_diff_std = data['price_diff'].std()
data = data[abs(data['price_diff']) <= 3 * price_diff_std]

# Drop rows with NaN values
data = data.dropna()

# Filter the data for training (2019-2023) and testing (2024)
train_data = data[(data['date'] >= '2001-01-01') & (data['date'] <= '2001-12-31')]
test_data = data[(data['date'] >= '2024-01-01') & (data['date'] <= '2024-12-31')]

# Define features (X) and target (y)
features = ['open', 'high', 'low', 'log_volume', 'lag_close_1', 'volatility_5', 'rsi_14', 'ma_deviation', 'day_of_week', 'month', 'year']
X_train = train_data[features]
y_train = train_data['price_diff']
X_test = test_data[features + ['close']]  # Include 'close' for price calculation
y_test = test_data['price_diff']
y_test_close = test_data['close']

# Scale the features (optional for Random Forest, kept for consistency)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test[features])

# Define Random Forest model
base_model = RandomForestRegressor(random_state=42)

# Perform grid search to tune hyperparameters
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
grid_search = GridSearchCV(
    base_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)

# Best model
model = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)

# Predict price differences
y_pred = model.predict(X_test_scaled)

# Convert to predicted close prices
y_pred_close = X_test['close'] + y_pred  # close = lag_close_1 + predicted_diff

# Calculate metrics for price differences
train_pred = model.predict(X_train_scaled)
train_mse = mean_squared_error(y_train, train_pred)
test_mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Train Mean Squared Error (Price Diff): {train_mse:.6f}")
print(f"Test Mean Squared Error (Price Diff): {test_mse:.6f}")
print(f"Test R² Score (Price Diff): {r2:.4f}")

# Calculate MSE for close prices
mse_close = mean_squared_error(y_test_close, y_pred_close)
print(f"Test Mean Squared Error (Close Prices): {mse_close:.6f}")

# Baseline: Predict previous day's close (for metrics only)
baseline_pred_close = test_data['lag_close_1']
baseline_mse_close = mean_squared_error(y_test_close, baseline_pred_close)
print(f"Baseline MSE (Close Prices, Previous Day): {baseline_mse_close:.6f}")

# Calculate directional accuracy
directional_accuracy = np.mean(np.sign(y_test) == np.sign(y_pred)) * 100
print(f"Directional Accuracy: {directional_accuracy:.2f}%")

# Feature importance
print("Feature Importance:")
for name, importance in zip(features, model.feature_importances_):
    print(f"{name}: {importance:.4f}")

# Plot 1: Actual vs. Predicted Close Prices
plt.figure(figsize=(12, 6))
plt.plot(test_data['date'], y_test_close, label='Actual Close Prices', color='blue', alpha=0.6)
plt.plot(test_data['date'], y_pred_close, label='Predicted Close Prices', color='red', alpha=0.6)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Random Forest Actual vs Predicted Close Prices (2024)')
plt.legend()
plt.grid()
plt.savefig('price_plot.png')
plt.close()

# Plot 2: Prediction Errors for Close Prices
errors = y_pred_close - y_test_close
plt.figure(figsize=(12, 6))
plt.plot(test_data['date'], errors, label='Prediction Errors (Predicted - Actual)', color='purple', alpha=0.6)
plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Error (Predicted - Actual Close Price)')
plt.title('Prediction Errors for Close Prices (2024)')
plt.legend()
plt.grid()
plt.savefig('error_plot.png')
plt.close()

# Plot 3: Actual vs. Predicted Price Differences
plt.figure(figsize=(12, 6))
plt.plot(test_data['date'], y_test, label='Actual Price Diff', color='blue', alpha=0.6)
plt.plot(test_data['date'], y_pred, label='Predicted Price Diff', color='red', alpha=0.6)
plt.xlabel('Date')
plt.ylabel('Price Difference (Close - Lag Close)')
plt.title('Random Forest Actual vs Predicted Price Differences (2024)')
plt.legend()
plt.grid()
plt.savefig('price_diff_plot.png')
plt.close()

# Plot 4: Directional Accuracy (Correctness Over Time)
correct_directions = (np.sign(y_test) == np.sign(y_pred)).astype(int)
plt.figure(figsize=(12, 6))
plt.plot(test_data['date'], correct_directions, label='Correct Direction (1=Correct, 0=Incorrect)', color='orange', alpha=0.6)
plt.xlabel('Date')
plt.ylabel('Correct Direction')
plt.title('Directional Prediction Correctness (2024)')
plt.legend()
plt.grid()
plt.savefig('directional_plot.png')
plt.close()

# Plot 5: Cumulative Returns from Model Predictions
# Convert price differences to returns for cumulative calculation
pred_returns = y_pred / test_data['lag_close_1']
cumulative_predicted_returns = np.cumprod(1 + pred_returns) - 1
plt.figure(figsize=(12, 6))
plt.plot(test_data['date'], cumulative_predicted_returns, label='Cumulative Returns (Model)', color='teal', alpha=0.6)
plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.title('Cumulative Returns from Model Predictions (2024)')
plt.legend()
plt.grid()
plt.savefig('cumulative_returns_plot.png')
plt.close()

# Classify predictions: Positive (1) if price_diff > 0, Negative (0) otherwise
y_test_class = (y_test > 0).astype(int)
y_pred_class = (y_pred > 0).astype(int)

# Compute confusion matrix
cm = confusion_matrix(y_test_class, y_pred_class)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix (Price Difference Classification)')
plt.savefig('confusion_matrix_plot.png')
plt.close()


# ROC Curve for Price Direction Prediction
fpr, tpr, thresholds = roc_curve(y_test_class, y_pred)  # Use raw predicted diffs as scores
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Price Direction Prediction (Random Forest)')
plt.legend(loc="lower right")
plt.grid()
plt.savefig('roc_curve_plot.png')
plt.close()
