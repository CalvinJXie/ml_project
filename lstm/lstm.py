import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Load the CSV file
data = pd.read_csv('KO_1919-09-06_2025-03-15.csv')

# Ensure the 'date' column is in datetime format with UTC
data['date'] = pd.to_datetime(data['date'], utc=True)

# Add derived features
data['log_volume'] = np.log1p(data['volume'])
data['lag_close_1'] = data['close'].shift(1)
data['volatility_5'] = data['close'].rolling(window=5).std()
data['price_diff'] = data['close'] - data['lag_close_1']

# Technical indicators
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
train_data = data[(data['date'] >= '2023-01-01') & (data['date'] <= '2023-12-31')]
test_data = data[(data['date'] >= '2024-01-01') & (data['date'] <= '2024-12-31')]

# Define features (X) and target (y)
features = ['open', 'high', 'low', 'log_volume', 'lag_close_1', 'volatility_5', 'rsi_14', 'ma_deviation', 'day_of_week', 'month', 'year']
X_train = train_data[features]
y_train = train_data['price_diff']
X_test = test_data[features]
y_test = test_data['price_diff']
y_test_close = test_data['close']

# Scale the features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)  # Add a time dimension
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return out

# Model parameters
input_size = X_train_tensor.shape[2]
hidden_size = 50
output_size = 1
model = LSTMModel(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 50
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

# Predict price differences
model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor).squeeze().numpy()

# Convert to predicted close prices
y_pred_close = test_data['lag_close_1'].values + y_pred  # close = lag_close_1 + predicted_diff

# Calculate metrics for price differences
train_pred = model(X_train_tensor).squeeze().detach().numpy()
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

# Plot 1: Actual vs. Predicted Close Prices
plt.figure(figsize=(12, 6))
plt.plot(test_data['date'], y_test_close, label='Actual Close Prices', color='blue', alpha=0.6)
plt.plot(test_data['date'], y_pred_close, label='Predicted Close Prices', color='red', alpha=0.6)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('PyTorch LSTM Actual vs Predicted Close Prices')
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
plt.title('Prediction Errors for Close Prices')
plt.legend()
plt.grid()
plt.savefig('error_plot.png')
plt.close()

# Plot 3: Confusion Matrix
y_test_class = (y_test > 0).astype(int)
y_pred_class = (y_pred > 0).astype(int)
cm = confusion_matrix(y_test_class, y_pred_class)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix (Price Difference Classification)')
plt.savefig('confusion_matrix_plot.png')
plt.close()