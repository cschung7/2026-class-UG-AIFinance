"""
03. LSTM (Long Short-Term Memory) for Time Series Forecasting
===============================================================
Goal: Use LSTM to predict stock prices with multi-feature input.

Key Concept:
  - LSTM solves RNN's vanishing gradient problem with 3 gates:
    - Forget gate:  f_t = sigmoid(W_f * [h_{t-1}, x_t])  -- what to forget
    - Input gate:   i_t = sigmoid(W_i * [h_{t-1}, x_t])  -- what to store
    - Output gate:  o_t = sigmoid(W_o * [h_{t-1}, x_t])  -- what to output
  - Cell state C_t carries long-term memory (like a conveyor belt)

Architecture:
  Input (seq_len, 5 features) -> LSTM(2 layers) -> FC -> Output
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
# 1. Download & Prepare Multi-Feature Data
# ============================================================
print("Downloading AAPL data with multiple features...")
df = yf.download("AAPL", start="2018-01-01", end="2025-12-31", auto_adjust=True)

# Use multiple features: Close, Volume, High-Low range, Open-Close change
df_feat = pd.DataFrame()
df_feat["Close"] = df["Close"].values.flatten()
df_feat["Volume"] = df["Volume"].values.flatten()
df_feat["HL_Range"] = (df["High"].values - df["Low"].values).flatten()
df_feat["OC_Change"] = (df["Close"].values - df["Open"].values).flatten()
df_feat["Return"] = df_feat["Close"].pct_change()
df_feat.dropna(inplace=True)

feature_cols = ["Close", "Volume", "HL_Range", "OC_Change", "Return"]
data = df_feat[feature_cols].values.astype(np.float32)

# Scale each feature independently
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# We predict Close price (index 0)
close_scaler = MinMaxScaler()
close_scaler.fit(df_feat[["Close"]].values.astype(np.float32))

SEQ_LEN = 30
BATCH_SIZE = 32
EPOCHS = 50


def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])           # (seq_len, n_features)
        y.append(data[i + seq_len, 0])             # Close price only
    return np.array(X), np.array(y).reshape(-1, 1)


X, y = create_sequences(data_scaled, SEQ_LEN)
print(f"Input shape: {X.shape} (samples, seq_len, features)")
print(f"Target shape: {y.shape}")

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

train_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
    batch_size=BATCH_SIZE, shuffle=True,
)
test_loader = DataLoader(
    TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
    batch_size=BATCH_SIZE, shuffle=False,
)


# ============================================================
# 2. LSTM Model
# ============================================================
class LSTMModel(nn.Module):
    """
    LSTM for multi-feature time series prediction.

    Internal flow per timestep:
      f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)   # Forget gate
      i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)   # Input gate
      c_t = f_t * c_{t-1} + i_t * tanh(W_c @ [h_{t-1}, x_t] + b_c)  # Cell state
      o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)   # Output gate
      h_t = o_t * tanh(c_t)                         # Hidden state
    """

    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # LSTM returns: output, (h_n, c_n)
        # h_n = final hidden state, c_n = final cell state
        out, (h_n, c_n) = self.lstm(x)
        out = out[:, -1, :]  # Last timestep
        out = self.fc(out)
        return out


# ============================================================
# 3. Train
# ============================================================
print("\n" + "=" * 60)
print("Training LSTM (2 layers, multi-feature)")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(input_size=5, hidden_size=64, num_layers=2).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    scheduler.step(avg_loss)
    if (epoch + 1) % 10 == 0:
        lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.6f}, LR: {lr:.6f}")

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

predictions = close_scaler.inverse_transform(np.array(predictions))
actuals = close_scaler.inverse_transform(np.array(actuals))

rmse = np.sqrt(mean_squared_error(actuals, predictions))
mae = mean_absolute_error(actuals, predictions)
mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
print(f"\nLSTM Results - RMSE: ${rmse:.2f}, MAE: ${mae:.2f}, MAPE: {mape:.2f}%")

# ---------- Plot ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("LSTM - Multi-Feature Stock Price Prediction (AAPL)", fontsize=14, fontweight="bold")

# (a) Training loss
axes[0, 0].plot(train_losses, color="#10b981", linewidth=1.5)
axes[0, 0].set_title("Training Loss")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("MSE Loss")
axes[0, 0].grid(True, alpha=0.3)

# (b) Full prediction
axes[0, 1].plot(actuals, label="Actual", color="#64748b", linewidth=1)
axes[0, 1].plot(predictions, label="LSTM", color="#10b981", linewidth=1, alpha=0.8)
axes[0, 1].set_title(f"Price Prediction (RMSE: ${rmse:.2f})")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# (c) Zoomed
n_zoom = 60
axes[1, 0].plot(actuals[-n_zoom:], label="Actual", color="#64748b", linewidth=1.5, marker="o", markersize=2)
axes[1, 0].plot(predictions[-n_zoom:], label="LSTM", color="#10b981", linewidth=1.5, marker="o", markersize=2)
axes[1, 0].set_title(f"Last {n_zoom} Days")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# (d) Scatter plot
axes[1, 1].scatter(actuals, predictions, alpha=0.3, s=10, color="#10b981")
min_val = min(actuals.min(), predictions.min())
max_val = max(actuals.max(), predictions.max())
axes[1, 1].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)
axes[1, 1].set_title(f"Actual vs Predicted (MAPE: {mape:.1f}%)")
axes[1, 1].set_xlabel("Actual Price ($)")
axes[1, 1].set_ylabel("Predicted Price ($)")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/mnt/nas/Class/2026/deeplearning/03_lstm_time_series.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 03_lstm_time_series.png")

# ============================================================
# 5. Visualize LSTM Gate Activations (Educational)
# ============================================================
print("\nLSTM Architecture Summary:")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"  LSTM layers: 2, Hidden size: 64")
print(f"  Input features: 5 (Close, Volume, HL_Range, OC_Change, Return)")
print(f"  Sequence length: {SEQ_LEN} days")

print("""
KEY INSIGHTS:
1. LSTM's cell state acts as long-term memory (forget/remember mechanism)
2. Multi-feature input (Volume, Range, etc.) gives richer context
3. Learning rate scheduler helps convergence
4. LSTM typically outperforms vanilla RNN on longer sequences
5. MAPE gives intuitive error interpretation (% off from actual price)
""")
