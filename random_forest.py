import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and prepare the data
df = pd.read_csv('KO_1919-09-06_2025-03-15.csv')
df.columns = df.columns.str.strip()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Create lag features and target
df['Lag_1'] = df['close'].shift(1)
df['Lag_2'] = df['close'].shift(2)
df['Lag_3'] = df['close'].shift(3)
df['Lag_4'] = df['close'].shift(4)
df['Lag_5'] = df['close'].shift(5)
df['Target'] = df['close'].shift(-1)

# Drop rows with NaNs
df = df.dropna()

# Define features and target
X = df[['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5']]
y = df['Target']

# Scale the features
scaler = StandardScaler()
#X_scaled = scaler.fit_transform(X)

# Split the data (no shuffling for time series)
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size=0.2, shuffle=False)

# Train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Random Forest MSE:", mse)
print("Random Forest MAE:", mae)

# Plot predictions vs actual
dates = df['date'].iloc[-len(y_test):]

plt.figure(figsize=(14, 6))
plt.plot(dates, y_test.values, label='Actual Price', alpha=0.7)
plt.plot(dates, y_pred, label='Random Forest Predicted Price', alpha=0.7)
plt.legend()
plt.title('KO Stock Price Prediction: Random Forest Regression')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot residuals
plt.figure(figsize=(14, 4))
plt.plot(dates, y_test.values - y_pred, label='Prediction Error (Residual)')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title('Random Forest Prediction Residuals: Actual - Predicted')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.tight_layout()
plt.show()

