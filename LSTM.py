import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Load and preprocess data
df = pd.read_csv('KO_1919-09-06_2025-03-15.csv')
df.columns = df.columns.str.strip()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Add engineered features
df['Open_Close_Change'] = (df['close'] - df['open']).clip(-5, 5)
df['High_Low_Range'] = (df['high'] - df['low']).clip(0, 10)
df['Volume_Change'] = df['volume'].pct_change().clip(-2, 2)
df['Volume_SMA_10'] = df['volume'].rolling(10).mean()
df['Volume_Volatility'] = df['volume'].rolling(10).std().rolling(5).mean()

# Drop rows with NaN values
df = df.dropna()

# Select features
features = ['close', 'Open_Close_Change', 'High_Low_Range', 'Volume_Change', 'Volume_SMA_10', 'Volume_Volatility']
data = df[['date'] + features].copy()
data.set_index('date', inplace=True)

# Normalize
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)
scaled = pd.DataFrame(scaled, columns=features, index=data.index)

# Sequence creator
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(data.iloc[i, 0])  # predict 'close'
    return np.array(X), np.array(y)

SEQ_LEN = 60
X, y = create_sequences(scaled, SEQ_LEN)

# Train-test split
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_size=X.shape[2]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
train_losses = []
for epoch in range(20):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")

# Plot training loss
plt.figure(figsize=(10, 4))
plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o')
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Predict
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor.to(device)).cpu().numpy()

# Inverse scale only the predicted 'close' value
predicted_close = scaler.inverse_transform(
    np.concatenate([predictions, X_test[:, -1, 1:]], axis=1))[:, 0]
actual_close = scaler.inverse_transform(
    np.concatenate([y_test_tensor.numpy(), X_test[:, -1, 1:]], axis=1))[:, 0]

# Plot predictions vs actual
dates = data.index[-len(actual_close):]
plt.figure(figsize=(14, 6))
plt.plot(dates, actual_close, label='Actual Price')
plt.plot(dates, predicted_close, label='LSTM Predicted Price')
plt.title('KO Stock Price Prediction: Enhanced LSTM with Engineered Features')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Residuals
plt.figure(figsize=(14, 4))
plt.plot(dates, actual_close - predicted_close, label='Prediction Error (Residual)')
plt.axhline(0, color='black', linestyle='--')
plt.title('LSTM Residuals: Actual - Predicted')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.tight_layout()
plt.show()

# Metrics
mse = mean_squared_error(actual_close, predicted_close)
mae = mean_absolute_error(actual_close, predicted_close)
print("Enhanced LSTM MSE:", mse)
print("Enhanced LSTM MAE:", mae)



