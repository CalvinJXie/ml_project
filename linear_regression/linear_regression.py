import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the CSV file
data = pd.read_csv('../KO_1919-09-06_2025-03-15.csv')

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

# Filter the data for training (2000-2008) and testing (2009-2010)
train_data = data[(data['date'] >= '2000-01-01') & (data['date'] <= '2001-12-31')]
test_data = data[(data['date'] >= '2024-01-01') & (data['date'] <= '2024-12-31')]

# Define features (X) and target (y)
features = ['open', 'high', 'low', 'log_volume', 'lag_close_1', 'volatility_5', 'rsi_14', 'ma_deviation', 'day_of_week', 'month', 'year']
X_train = train_data[features]
y_train = train_data['price_diff']
X_test = test_data[features + ['close']]  # Include 'close' for price calculation
y_test = test_data['price_diff']
y_test_close = test_data['close']

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test[features])

# Define Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train_scaled, y_train)

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

# Baseline: Predict previous day's close
baseline_pred_close = test_data['lag_close_1']
baseline_mse_close = mean_squared_error(y_test_close, baseline_pred_close)
print(f"Baseline MSE (Close Prices, Previous Day): {baseline_mse_close:.6f}")

# Calculate directional accuracy
directional_accuracy = np.mean(np.sign(y_test) == np.sign(y_pred)) * 100
print(f"Directional Accuracy: {directional_accuracy:.2f}%")

# Feature coefficients
print("Feature Coefficients:")
for name, coef in zip(features, model.coef_):
    print(f"{name}: {coef:.4f}")

# Plot 1: Actual vs. Predicted Close Prices with Baseline
plt.figure(figsize=(12, 6))
plt.plot(test_data['date'], y_test_close, label='Actual Close Prices', color='blue', alpha=0.6)
plt.plot(test_data['date'], y_pred_close, label='Predicted Close Prices', color='red', alpha=0.6)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Actual vs Predicted Close Prices vs Baseline (2009-2010)')
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
plt.title('Prediction Errors for Close Prices (2009-2010)')
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
plt.title('Actual vs Predicted Price Differences (2009-2010)')
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
plt.title('Directional Prediction Correctness (2009-2010)')
plt.legend()
plt.grid()
plt.savefig('directional_plot.png')
plt.close()

# Plot 5: Cumulative Excess Returns
# Convert price differences to returns for cumulative calculation
actual_returns = y_test / test_data['lag_close_1']
pred_returns = y_pred / test_data['lag_close_1']
cumulative_actual_returns = np.cumprod(1 + actual_returns) - 1
cumulative_predicted_returns = np.cumprod(1 + pred_returns) - 1
# Baseline: Assume zero price difference (close = lag_close_1)
baseline_returns = np.zeros_like(y_test)
cumulative_baseline_returns = np.cumprod(1 + baseline_returns) - 1
excess_returns = cumulative_predicted_returns - cumulative_baseline_returns
plt.figure(figsize=(12, 6))
plt.plot(test_data['date'], excess_returns, label='Excess Returns (Model - Baseline)', color='teal', alpha=0.6)
plt.axhline(0, color='black', linestyle='--', alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Cumulative Excess Returns')
plt.title('Cumulative Excess Returns (Model vs Baseline, 2009-2010)')
plt.legend()
plt.grid()
plt.savefig('excess_returns_plot.png')
plt.close()