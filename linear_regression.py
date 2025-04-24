
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Load and prepare the data
df = pd.read_csv('KO_1919-09-06_2025-03-15.csv')
print(df.columns)
df.columns = df.columns.str.strip()
df['date'] = pd.to_datetime(df['date'], utc = True)
df = df.sort_values('date')

# Additional engineered features
df['Open_Close_Change'] = df['close'] - df['open']
df['High_Low_Range'] = df['high'] - df['low']
df['Volume_Change'] = df['volume'].pct_change(1)
df['Volume_SMA_10'] = df['volume'].rolling(10).mean()
df['Volume_Volatility'] = df['volume'].rolling(10).std()

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
feature_cols = ['Lag_1', 'Lag_2', 'Lag_3', 'Lag_4', 'Lag_5',
                'Open_Close_Change', 'High_Low_Range', 
                'Volume_Change', 'Volume_SMA_10', 'Volume_Volatility']
X = df[feature_cols]
y = df['Target']

# Split the data (no shuffling for time series)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)

# Plot predictions vs actual
dates = df['date'].iloc[-len(y_test):]

plt.figure(figsize=(14, 6))
plt.plot(dates, y_test.values, label='Actual Price')
plt.plot(dates, y_pred, label='Predicted Price')
plt.legend()
plt.title('KO Stock Price Prediction: Linear Regression (2000–2025)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 4))
plt.plot(dates, y_test.values - y_pred, label='Prediction Error (Residual)')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title('Prediction Residuals: Actual - Predicted')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.tight_layout()
plt.show()

# --- Load new data for out-of-sample testing (after 2025-03-15) ---
new_df = pd.read_csv('KO_1919-09-06_2025-04-17.csv')
new_df.columns = new_df.columns.str.strip()
new_df['date'] = pd.to_datetime(new_df['date'], utc = True)
new_df = new_df.sort_values('date')
new_df = new_df[new_df['date'] > pd.Timestamp('2025-03-15', tz='UTC')]


# Reapply feature engineering
new_df['Open_Close_Change'] = new_df['close'] - new_df['open']
new_df['High_Low_Range'] = new_df['high'] - new_df['low']
new_df['Volume_Change'] = new_df['volume'].pct_change(1)
new_df['Volume_SMA_10'] = new_df['volume'].rolling(10).mean()
new_df['Volume_Volatility'] = new_df['volume'].rolling(10).std()
new_df['Lag_1'] = new_df['close'].shift(1)
new_df['Lag_2'] = new_df['close'].shift(2)
new_df['Lag_3'] = new_df['close'].shift(3)
new_df['Lag_4'] = new_df['close'].shift(4)
new_df['Lag_5'] = new_df['close'].shift(5)
new_df['Target'] = new_df['close'].shift(-1)

new_df = new_df.dropna()

# Features for prediction
X_new = new_df[feature_cols]
y_actual = new_df['Target']
dates_new = new_df['date']

# Predict using the trained model
y_new_pred = model.predict(X_new)

# --- Plot new prediction chart ---
plt.figure(figsize=(14, 6))
plt.plot(dates_new, y_actual, label='Actual Price')
plt.plot(dates_new, y_new_pred, label='Predicted Price')
plt.legend()
plt.title('Out-of-Sample Prediction (2025-03-16 to 2025-04-17)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


