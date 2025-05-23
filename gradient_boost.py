import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# Load and prepare the data
df = pd.read_csv('KO_1919-09-06_2025-03-15.csv')
df.columns = df.columns.str.strip()
df['date'] = pd.to_datetime(df['date'], utc=True)
df = df[df['date'] >= pd.Timestamp('2000-01-01', tz='UTC')]

# Lag features
for lag in range(1, 31):
    df[f'Lag_{lag}'] = df['close'].shift(lag)

# Moving averages
df['SMA_10'] = df['close'].rolling(10).mean()
df['SMA_30'] = df['close'].rolling(30).mean()

# Volatility
df['Volatility'] = df['close'].rolling(10).std()

# Additional features from raw columns
df['Open_Close_Change'] = df['close'] - df['open']
df['High_Low_Range'] = df['high'] - df['low']
df['Volume_Change'] = df['volume'].pct_change(1)
df['Volume_SMA_10'] = df['volume'].rolling(10).mean()
df['Volume_Volatility'] = df['volume'].rolling(10).std()

# Rate of change (returns)
df['Return_1d'] = df['close'].pct_change(1)
df['Return_5d'] = df['close'].pct_change(5)
df['Return_10d'] = df['close'].pct_change(10)

# Momentum
df['Momentum_10'] = df['close'] - df['close'].shift(10)

# Target: 5-day future price
df['Target'] = df['close'].shift(-5)

# Drop NaN rows
df = df.dropna()

# Feature selection
feature_cols = [col for col in df.columns if col in ['Open_Close_Change', 'High_Low_Range', 'Volume_Change', 'Volume_SMA_10', 'Volume_Volatility'] or col.startswith('Lag_') or 
                'SMA' in col or 'Volatility' in col or 
                'Return' in col or 'Momentum' in col]

X = df[feature_cols]
y = df['Target']

# Standardize all features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Gradient Boosting model on full dataset
gbr_model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=7,
    subsample=0.9,
    max_features=0.8,
    random_state=42
)
gbr_model.fit(X_scaled, y)

# Predict on full data
y_pred_all = gbr_model.predict(X_scaled)

# Evaluation on full set
mse = mean_squared_error(y, y_pred_all)
mae = mean_absolute_error(y, y_pred_all)
print("Gradient Boosting MSE:", mse)
print("Gradient Boosting MAE:", mae)
print("Target Range:", y.min(), "-", y.max())
print("Predicted Range:", y_pred_all.min(), "-", y_pred_all.max())
print("Unique Predicted Values (rounded):", len(np.unique(np.round(y_pred_all, 2))))

# Plot prediction vs actual for full date range
plt.figure(figsize=(14, 6))
plt.plot(df['date'], y.values, label='Actual Price', alpha=0.7)
plt.plot(df['date'], y_pred_all, label='GBR Predicted Price', alpha=0.7)
plt.title('KO Stock Price Prediction: Gradient Boosting (2000–2025)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot residuals
plt.figure(figsize=(14, 4))
plt.plot(df['date'], y.values - y_pred_all, label='Prediction Error (Residual)')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title('GBR Residuals: Actual - Predicted')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.tight_layout()
plt.show()

# Feature importance
importances = gbr_model.feature_importances_
indices = np.argsort(importances)[-15:][::-1]
plt.figure(figsize=(10, 6))
plt.bar(range(len(indices)), importances[indices])
plt.xticks(range(len(indices)), [feature_cols[i] for i in indices], rotation=45)
plt.title("Gradient Boosting Feature Importances")
plt.tight_layout()
plt.show()

# ==========================
# Future Data Prediction
# ==========================

# Load full CSV that includes future dates
df_future = pd.read_csv('KO_1919-09-06_2025-04-17.csv')
df_future.columns = df_future.columns.str.strip()
df_future['date'] = pd.to_datetime(df_future['date'], utc=True)

# Feature Engineering
for lag in range(1, 31):
    df_future[f'Lag_{lag}'] = df_future['close'].shift(lag)
df_future['SMA_10'] = df_future['close'].rolling(10).mean()
df_future['SMA_30'] = df_future['close'].rolling(30).mean()
df_future['Volatility'] = df_future['close'].rolling(10).std()
df_future['Open_Close_Change'] = df_future['close'] - df_future['open']
df_future['High_Low_Range'] = df_future['high'] - df_future['low']
df_future['Volume_Change'] = df_future['volume'].pct_change(1)
df_future['Volume_SMA_10'] = df_future['volume'].rolling(10).mean()
df_future['Volume_Volatility'] = df_future['volume'].rolling(10).std()
df_future['Return_1d'] = df_future['close'].pct_change(1)
df_future['Return_5d'] = df_future['close'].pct_change(5)
df_future['Return_10d'] = df_future['close'].pct_change(10)
df_future['Momentum_10'] = df_future['close'] - df_future['close'].shift(10)
df_future['Target'] = df_future['close'].shift(-5)

# Drop rows with NaNs
df_future = df_future.dropna()

# Only keep rows after 2025-03-15
df_future = df_future[df_future['date'] > pd.Timestamp('2025-03-15', tz='UTC')]

# Prepare feature matrix and target
X_future = df_future[feature_cols]
y_future = df_future['Target']
X_future_scaled = scaler.transform(X_future)

# Predict
y_future_pred = gbr_model.predict(X_future_scaled)

# Evaluation
print("\nFuture Data Evaluation (After 2025-03-15):")
print("MSE:", mean_squared_error(y_future, y_future_pred))
print("MAE:", mean_absolute_error(y_future, y_future_pred))

# Plot prediction vs actual
plt.figure(figsize=(14, 6))
plt.plot(df_future['date'], y_future.values, label='Actual Future Price', alpha=0.7)
plt.plot(df_future['date'], y_future_pred, label='Predicted Future Price', alpha=0.7)
plt.title('KO Stock Prediction: 2025-03-16 to 2025-04-17')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


