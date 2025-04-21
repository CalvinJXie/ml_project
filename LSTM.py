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

data = df[['date', 'close']].copy()
data.set_index('date', inplace=True)

scaler = MinMaxScaler()
data['close'] = scaler.fit_transform(data[['close']])
close_values = data['close'].values

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

SEQ_LEN = 60
X, y = create_sequences(close_values, SEQ_LEN)

# Train-test split
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train with loss tracking
train_losses = []
for epoch in range(20):
    total_loss = 0
    model.train()
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

predicted_prices = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(y_test_tensor.numpy())

# Plot predicted vs actual
dates = data.index[-len(actual_prices):]
plt.figure(figsize=(14, 6))
plt.plot(dates, actual_prices, label='Actual Price')
plt.plot(dates, predicted_prices, label='LSTM Predicted Price')
plt.title('KO Stock Price Prediction: PyTorch LSTM')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Residual plot
plt.figure(figsize=(14, 4))
plt.plot(dates, actual_prices.flatten() - predicted_prices.flatten(), label='Prediction Error (Residual)')
plt.axhline(0, color='black', linestyle='--')
plt.title('LSTM Prediction Residuals: Actual - Predicted')
plt.xlabel('Date')
plt.ylabel('Error')
plt.legend()
plt.tight_layout()
plt.show()

# Metrics
mse = mean_squared_error(actual_prices, predicted_prices)
mae = mean_absolute_error(actual_prices, predicted_prices)
print("PyTorch LSTM MSE:", mse)
print("PyTorch LSTM MAE:", mae)


