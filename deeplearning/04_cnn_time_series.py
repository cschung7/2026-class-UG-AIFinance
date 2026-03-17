"""
04. CNN (Convolutional Neural Network) for Time Series Forecasting
====================================================================
Goal: Use 1D CNN to detect local patterns in stock price sequences.

Key Concept:
  - CNN slides a filter (kernel) across the time series
  - Each filter detects a specific local pattern (e.g., V-shape recovery, trend)
  - Multiple filters = multiple pattern detectors
  - Pooling reduces dimensionality while keeping important features

Why CNN for Time Series?
  - Fast training (parallelizable, unlike RNN)
  - Captures local patterns (3-day, 5-day, 7-day patterns)
  - Translation invariant: same pattern detected anywhere in sequence

Architecture:
  Input (seq_len, features) -> Conv1D -> Pool -> Conv1D -> Pool -> FC -> Output
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
print("Downloading MSFT data...")
df = yf.download("MSFT", start="2018-01-01", end="2025-12-31", auto_adjust=True)

df_feat = pd.DataFrame()
df_feat["Close"] = df["Close"].values.flatten()
df_feat["Volume"] = df["Volume"].values.flatten()
df_feat["HL_Range"] = (df["High"].values - df["Low"].values).flatten()
df_feat["OC_Change"] = (df["Close"].values - df["Open"].values).flatten()
df_feat["Return"] = df_feat["Close"].pct_change()
df_feat.dropna(inplace=True)

feature_cols = ["Close", "Volume", "HL_Range", "OC_Change", "Return"]
data = df_feat[feature_cols].values.astype(np.float32)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

close_scaler = MinMaxScaler()
close_scaler.fit(df_feat[["Close"]].values.astype(np.float32))

SEQ_LEN = 30
BATCH_SIZE = 32
EPOCHS = 50
N_FEATURES = len(feature_cols)


def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len, 0])
    return np.array(X), np.array(y).reshape(-1, 1)


X, y = create_sequences(data_scaled, SEQ_LEN)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# CNN expects (batch, channels, length) -> transpose features to channel dim
X_train_t = torch.FloatTensor(X_train).permute(0, 2, 1)  # (batch, features, seq_len)
X_test_t = torch.FloatTensor(X_test).permute(0, 2, 1)
y_train_t = torch.FloatTensor(y_train)
y_test_t = torch.FloatTensor(y_test)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False
)
print(f"CNN Input shape: {X_train_t.shape} (batch, channels/features, seq_len)")


# ============================================================
# 2. 1D CNN Model
# ============================================================
class CNN1D(nn.Module):
    """
    1D CNN for time series.

    How it works:
    1. Conv1D slides filters across time dimension
       - kernel_size=3: looks at 3 consecutive timesteps
       - Each filter learns to detect a specific pattern
    2. MaxPool1D reduces sequence length (keeps strongest signals)
    3. Multiple conv layers = hierarchical pattern detection
       - Layer 1: short patterns (3-day)
       - Layer 2: medium patterns (5-7 day, built from layer 1)
    """

    def __init__(self, n_features=5, seq_len=30):
        super().__init__()

        self.conv_block = nn.Sequential(
            # Layer 1: 32 filters, kernel=3 -> detects 3-day patterns
            nn.Conv1d(in_channels=n_features, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # seq_len: 30 -> 15

            # Layer 2: 64 filters, kernel=3 -> detects ~6-day patterns
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # seq_len: 15 -> 7

            # Layer 3: 128 filters, kernel=3 -> detects ~14-day patterns
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # Global avg pool -> (batch, 128, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x shape: (batch, n_features, seq_len)
        out = self.conv_block(x)     # (batch, 128, 1)
        out = out.squeeze(-1)         # (batch, 128)
        out = self.fc(out)            # (batch, 1)
        return out


# ============================================================
# 3. Train
# ============================================================
print("\n" + "=" * 60)
print("Training 1D CNN")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN1D(n_features=N_FEATURES, seq_len=SEQ_LEN).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

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
predictions, actuals = [], []
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
print(f"\nCNN Results - RMSE: ${rmse:.2f}, MAE: ${mae:.2f}, MAPE: {mape:.2f}%")

# ---------- Visualize Filters ----------
# Extract first conv layer filters
filters = model.conv_block[0].weight.data.cpu().numpy()  # (32, 5, 3)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("1D CNN - Time Series Prediction (MSFT)", fontsize=14, fontweight="bold")

# (a) Training loss
axes[0, 0].plot(train_losses, color="#f59e0b", linewidth=1.5)
axes[0, 0].set_title("Training Loss")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("MSE Loss")
axes[0, 0].grid(True, alpha=0.3)

# (b) Predictions
axes[0, 1].plot(actuals, label="Actual", color="#64748b", linewidth=1)
axes[0, 1].plot(predictions, label="CNN", color="#f59e0b", linewidth=1, alpha=0.8)
axes[0, 1].set_title(f"Price Prediction (RMSE: ${rmse:.2f})")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# (c) Zoomed
n_zoom = 60
axes[0, 2].plot(actuals[-n_zoom:], label="Actual", color="#64748b", linewidth=1.5)
axes[0, 2].plot(predictions[-n_zoom:], label="CNN", color="#f59e0b", linewidth=1.5)
axes[0, 2].set_title(f"Last {n_zoom} Days")
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# (d) Conv filter visualization (first 8 filters)
for i in range(8):
    row = i // 4
    col = i % 4
    if i < 4:
        ax_idx = axes[1, 0]
    else:
        ax_idx = axes[1, 1]

# Show first 8 learned filters as a heatmap
filter_data = filters[:8, :, :].reshape(8, -1)  # (8 filters, 5*3)
im = axes[1, 0].imshow(filter_data, aspect="auto", cmap="RdBu_r")
axes[1, 0].set_title("Learned Conv Filters (first 8)")
axes[1, 0].set_xlabel("Weight Index")
axes[1, 0].set_ylabel("Filter #")
plt.colorbar(im, ax=axes[1, 0])

# (e) Error distribution
errors = (predictions - actuals).flatten()
axes[1, 1].hist(errors, bins=50, color="#f59e0b", alpha=0.7, edgecolor="black")
axes[1, 1].axvline(0, color="red", linestyle="--")
axes[1, 1].set_title(f"Error Distribution (MAE: ${mae:.2f})")
axes[1, 1].set_xlabel("Error ($)")
axes[1, 1].grid(True, alpha=0.3)

# (f) Scatter
axes[1, 2].scatter(actuals, predictions, alpha=0.3, s=10, color="#f59e0b")
mn, mx = min(actuals.min(), predictions.min()), max(actuals.max(), predictions.max())
axes[1, 2].plot([mn, mx], [mn, mx], "r--")
axes[1, 2].set_title(f"Actual vs Predicted (MAPE: {mape:.1f}%)")
axes[1, 2].set_xlabel("Actual ($)")
axes[1, 2].set_ylabel("Predicted ($)")
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/mnt/nas/Class/2026/deeplearning/04_cnn_time_series.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 04_cnn_time_series.png")

print("""
KEY INSIGHTS:
1. CNN processes the entire sequence in parallel (faster than RNN)
2. Each filter is a local pattern detector (3-day window)
3. Stacking conv layers captures increasingly complex patterns
4. MaxPool keeps the strongest signals, reduces computation
5. AdaptiveAvgPool makes the model flexible to different input lengths
6. CNN works well for detecting regime changes and local trends
""")
