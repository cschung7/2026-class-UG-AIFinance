"""
02. RNN (Recurrent Neural Network) for Time Series Forecasting
================================================================
Goal: Use vanilla RNN to predict stock prices.

Key Concept:
  - RNN processes sequences step-by-step, maintaining a hidden state
  - h_t = tanh(W_hh * h_{t-1} + W_xh * x_t + b)
  - Good for sequential data but suffers from vanishing gradients

Architecture:
  Input (seq_len, features) -> RNN -> FC -> Output (1 price)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ============================================================
# 1. Download & Prepare Data
# ============================================================
print("Downloading AAPL data...")
df = yf.download("AAPL", start="2018-01-01", end="2025-12-31", auto_adjust=True)
data = df[["Close"]].values.astype(np.float32)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences: use past SEQ_LEN days to predict next day
SEQ_LEN = 30
BATCH_SIZE = 32
EPOCHS = 50


def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


X, y = create_sequences(data_scaled, SEQ_LEN)
print(f"Sequences created: X={X.shape}, y={y.shape}")
# X shape: (samples, seq_len, 1), y shape: (samples, 1)

# Train/Test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
# 2. Define Vanilla RNN Model
# ============================================================
class VanillaRNN(nn.Module):
    """
    Simple RNN for time series prediction.

    How it works:
    1. Input sequence passes through RNN cell one timestep at a time
    2. Each timestep updates the hidden state: h_t = tanh(W * [h_{t-1}, x_t])
    3. Final hidden state is passed through a fully connected layer
    """

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,      # Input shape: (batch, seq_len, features)
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        # h0 shape: (num_layers, batch, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.rnn(x, h0)   # out: (batch, seq_len, hidden_size)
        out = out[:, -1, :]         # Take last timestep: (batch, hidden_size)
        out = self.fc(out)          # (batch, 1)
        return out


# ============================================================
# 3. Train
# ============================================================
print("\n" + "=" * 60)
print("Training Vanilla RNN")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VanillaRNN(input_size=1, hidden_size=64, num_layers=2).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses = []
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}")

# ============================================================
# 4. Evaluate & Plot
# ============================================================
model.eval()
predictions = []
actuals = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        pred = model(xb).cpu().numpy()
        predictions.extend(pred)
        actuals.extend(yb.numpy())

predictions = scaler.inverse_transform(np.array(predictions))
actuals = scaler.inverse_transform(np.array(actuals))

rmse = np.sqrt(mean_squared_error(actuals, predictions))
mae = mean_absolute_error(actuals, predictions)
print(f"\nRNN Results - RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# ---------- Plot ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Vanilla RNN - Stock Price Prediction (AAPL)", fontsize=14, fontweight="bold")

# (a) Training loss
axes[0, 0].plot(train_losses, color="#3b82f6", linewidth=1.5)
axes[0, 0].set_title("Training Loss")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("MSE Loss")
axes[0, 0].grid(True, alpha=0.3)

# (b) Full prediction vs actual
axes[0, 1].plot(actuals, label="Actual", color="#64748b", linewidth=1)
axes[0, 1].plot(predictions, label="RNN Predicted", color="#3b82f6", linewidth=1, alpha=0.8)
axes[0, 1].set_title(f"Price Prediction (RMSE: {rmse:.2f})")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# (c) Zoomed last 60 days
n_zoom = 60
axes[1, 0].plot(actuals[-n_zoom:], label="Actual", color="#64748b", linewidth=1.5)
axes[1, 0].plot(predictions[-n_zoom:], label="RNN", color="#3b82f6", linewidth=1.5, alpha=0.8)
axes[1, 0].set_title(f"Last {n_zoom} Days (Zoomed)")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# (d) Prediction error distribution
errors = (predictions - actuals).flatten()
axes[1, 1].hist(errors, bins=50, color="#3b82f6", alpha=0.7, edgecolor="black")
axes[1, 1].axvline(0, color="red", linestyle="--", linewidth=1)
axes[1, 1].set_title(f"Error Distribution (MAE: {mae:.2f})")
axes[1, 1].set_xlabel("Prediction Error ($)")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/mnt/nas/Class/2026/deeplearning/02_rnn_time_series.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 02_rnn_time_series.png")

print("""
KEY INSIGHTS:
1. Vanilla RNN struggles with long sequences (vanishing gradient problem)
2. Gradient clipping helps stabilize training
3. RNN captures short-term patterns but misses long-term dependencies
4. Next: LSTM solves the vanishing gradient problem with gates
""")
