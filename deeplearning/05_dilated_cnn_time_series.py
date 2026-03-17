"""
05. Dilated CNN (WaveNet-style) for Time Series Forecasting
==============================================================
Goal: Use dilated convolutions to capture long-range dependencies
      without pooling or recurrence.

Key Concept:
  - Regular Conv1D with kernel=3 sees 3 timesteps
  - Dilated Conv1D with kernel=3, dilation=2 sees 3 timesteps spread over 5
  - Stacking dilations exponentially: d=1,2,4,8,16
    -> Receptive field grows exponentially: 3, 7, 15, 31, 63 timesteps!

  Regular:    [x] [x] [x]  .   .   .   .    (sees 3 timesteps)
  Dilation=2: [x]  .  [x]  .  [x]  .   .    (sees 5 timesteps)
  Dilation=4: [x]  .   .   .  [x]  .   .   .  [x]  (sees 9 timesteps)

Why Dilated CNN?
  - Exponential receptive field without losing resolution
  - Parallelizable (unlike RNN/LSTM)
  - No vanishing gradients (unlike RNN)
  - Used in WaveNet (Google), TCN, and many state-of-the-art models

Architecture:
  Input -> [DilatedBlock(d=1) -> DilatedBlock(d=2) -> DilatedBlock(d=4) -> ...] -> FC -> Output
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
print("Downloading NVDA data...")
df = yf.download("NVDA", start="2018-01-01", end="2025-12-31", auto_adjust=True)

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

SEQ_LEN = 60  # Longer sequence to show dilated CNN advantage
BATCH_SIZE = 32
EPOCHS = 60
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

# CNN: (batch, channels, seq_len)
X_train_t = torch.FloatTensor(X_train).permute(0, 2, 1)
X_test_t = torch.FloatTensor(X_test).permute(0, 2, 1)
y_train_t = torch.FloatTensor(y_train)
y_test_t = torch.FloatTensor(y_test)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(X_test_t, y_test_t), batch_size=BATCH_SIZE, shuffle=False
)


# ============================================================
# 2. Dilated Causal Conv Block
# ============================================================
class DilatedCausalConvBlock(nn.Module):
    """
    Single dilated causal convolution block with residual connection.

    Causal: uses left-padding so output at time t only depends on t and before.
    Residual: skip connection helps gradient flow (like ResNet).

    Architecture:
      Input -> CausalPad -> DilatedConv -> ReLU -> Dropout -> Conv1x1 -> + Input -> ReLU
    """

    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation  # Causal padding

        self.dilated_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,  # We'll pad manually for causal
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.conv1x1 = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        # Causal padding: pad only on the left side
        padded = nn.functional.pad(x, (self.padding, 0))
        out = self.dilated_conv(padded)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv1x1(out)
        return self.activation(out + x)  # Residual connection


class DilatedCNN(nn.Module):
    """
    WaveNet-style dilated CNN for time series.

    Dilations: [1, 2, 4, 8, 16] -> receptive field covers 63 timesteps!

    Receptive Field Calculation:
      R = 1 + sum(kernel_size - 1) * dilation for each layer
      R = 1 + (3-1)*1 + (3-1)*2 + (3-1)*4 + (3-1)*8 + (3-1)*16
      R = 1 + 2 + 4 + 8 + 16 + 32 = 63 timesteps
    """

    def __init__(self, n_features=5, hidden_channels=64, kernel_size=3, dilations=None):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8, 16]  # Exponential growth

        # Input projection
        self.input_conv = nn.Conv1d(n_features, hidden_channels, kernel_size=1)

        # Dilated blocks
        self.blocks = nn.ModuleList([
            DilatedCausalConvBlock(
                channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=d,
            )
            for d in dilations
        ])

        # Output
        self.output = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

        # Calculate receptive field
        self.receptive_field = 1 + sum((kernel_size - 1) * d for d in dilations)

    def forward(self, x):
        # x: (batch, n_features, seq_len)
        out = self.input_conv(x)  # Project to hidden_channels

        for block in self.blocks:
            out = block(out)

        out = self.output(out)
        return out


# ============================================================
# 3. Also build a Regular CNN for comparison
# ============================================================
class RegularCNN(nn.Module):
    """Regular CNN (no dilation) for comparison."""
    def __init__(self, n_features=5, hidden_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 4. Train Both Models
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, epochs, name):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []
    print(f"\nTraining {name}...")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    if hasattr(model, "receptive_field"):
        print(f"  Receptive field: {model.receptive_field} timesteps")

    for epoch in range(epochs):
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
        losses.append(avg_loss)
        if (epoch + 1) % 15 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    return losses


def evaluate_model(model, test_loader):
    model.eval()
    preds, acts = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()
            preds.extend(pred)
            acts.extend(yb.numpy())
    preds = close_scaler.inverse_transform(np.array(preds))
    acts = close_scaler.inverse_transform(np.array(acts))
    return preds, acts


print("=" * 60)
print("Training Dilated CNN vs Regular CNN")
print("=" * 60)

model_dilated = DilatedCNN(n_features=N_FEATURES, hidden_channels=64, dilations=[1, 2, 4, 8, 16])
model_regular = RegularCNN(n_features=N_FEATURES)

losses_dilated = train_model(model_dilated, train_loader, EPOCHS, "Dilated CNN")
losses_regular = train_model(model_regular, train_loader, EPOCHS, "Regular CNN")

preds_dilated, actuals = evaluate_model(model_dilated, test_loader)
preds_regular, _ = evaluate_model(model_regular, test_loader)

# Metrics
results = {}
for name, preds in [("Dilated CNN", preds_dilated), ("Regular CNN", preds_regular)]:
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae = mean_absolute_error(actuals, preds)
    mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
    results[name] = {"RMSE": rmse, "MAE": mae, "MAPE": mape}
    print(f"\n{name} - RMSE: ${rmse:.2f}, MAE: ${mae:.2f}, MAPE: {mape:.2f}%")

# ============================================================
# 5. Comprehensive Plots
# ============================================================
fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle("Dilated CNN vs Regular CNN - Time Series (NVDA)", fontsize=14, fontweight="bold")

# (a) Training loss comparison
axes[0, 0].plot(losses_regular, label="Regular CNN", color="#94a3b8", linewidth=1.5)
axes[0, 0].plot(losses_dilated, label="Dilated CNN", color="#8b5cf6", linewidth=1.5)
axes[0, 0].set_title("Training Loss Comparison")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# (b) Full predictions comparison
axes[0, 1].plot(actuals, label="Actual", color="#64748b", linewidth=1)
axes[0, 1].plot(preds_regular, label="Regular CNN", color="#94a3b8", linewidth=1, alpha=0.7)
axes[0, 1].plot(preds_dilated, label="Dilated CNN", color="#8b5cf6", linewidth=1, alpha=0.8)
axes[0, 1].set_title("Price Predictions")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# (c) Zoomed comparison
n_zoom = 60
axes[1, 0].plot(actuals[-n_zoom:], label="Actual", color="#64748b", linewidth=2)
axes[1, 0].plot(preds_regular[-n_zoom:], label="Regular", color="#94a3b8", linewidth=1.5)
axes[1, 0].plot(preds_dilated[-n_zoom:], label="Dilated", color="#8b5cf6", linewidth=1.5)
axes[1, 0].set_title(f"Last {n_zoom} Days (Zoomed)")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# (d) Receptive field visualization
ax = axes[1, 1]
dilations = [1, 2, 4, 8, 16]
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(dilations)))
for i, (d, c) in enumerate(zip(dilations, colors)):
    # Show which timesteps each layer "sees"
    positions = [0, d, 2 * d]  # 3 kernel positions with dilation
    ax.scatter(positions, [i] * 3, color=c, s=100, zorder=5)
    ax.plot(positions, [i] * 3, color=c, linewidth=2)
ax.set_yticks(range(len(dilations)))
ax.set_yticklabels([f"d={d}" for d in dilations])
ax.set_xlabel("Timestep offset")
ax.set_title("Dilated Conv Receptive Field")
ax.grid(True, alpha=0.3)

# (e) Error comparison
errors_dilated = (preds_dilated - actuals).flatten()
errors_regular = (preds_regular - actuals).flatten()
axes[2, 0].hist(errors_regular, bins=50, alpha=0.5, color="#94a3b8", label="Regular", edgecolor="black")
axes[2, 0].hist(errors_dilated, bins=50, alpha=0.5, color="#8b5cf6", label="Dilated", edgecolor="black")
axes[2, 0].axvline(0, color="red", linestyle="--")
axes[2, 0].set_title("Error Distribution")
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# (f) Metrics bar chart
names = list(results.keys())
mapes = [results[n]["MAPE"] for n in names]
maes = [results[n]["MAE"] for n in names]
x = np.arange(len(names))
width = 0.35
ax2 = axes[2, 1]
bars1 = ax2.bar(x - width / 2, mapes, width, label="MAPE (%)", color="#8b5cf6", alpha=0.7)
ax2_twin = ax2.twinx()
bars2 = ax2_twin.bar(x + width / 2, maes, width, label="MAE ($)", color="#f59e0b", alpha=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels(names)
ax2.set_ylabel("MAPE (%)")
ax2_twin.set_ylabel("MAE ($)")
ax2.set_title("Model Comparison")
ax2.legend(loc="upper left")
ax2_twin.legend(loc="upper right")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/mnt/nas/Class/2026/deeplearning/05_dilated_cnn_time_series.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nSaved: 05_dilated_cnn_time_series.png")

# ============================================================
# 6. Summary
# ============================================================
print("\n" + "=" * 60)
print("MODEL COMPARISON SUMMARY")
print("=" * 60)

regular_rf = 1 + (3 - 1) * 5  # 5 layers, kernel=3, dilation=1
dilated_rf = model_dilated.receptive_field
print(f"""
{'Model':<20} {'RMSE':>8} {'MAE':>8} {'MAPE':>8} {'Receptive Field':>16} {'Params':>10}
{'='*70}
{'Regular CNN':<20} ${results['Regular CNN']['RMSE']:>6.2f} ${results['Regular CNN']['MAE']:>6.2f} {results['Regular CNN']['MAPE']:>6.2f}% {regular_rf:>12} steps {sum(p.numel() for p in model_regular.parameters()):>10,}
{'Dilated CNN':<20} ${results['Dilated CNN']['RMSE']:>6.2f} ${results['Dilated CNN']['MAE']:>6.2f} {results['Dilated CNN']['MAPE']:>6.2f}% {dilated_rf:>12} steps {sum(p.numel() for p in model_dilated.parameters()):>10,}

KEY INSIGHTS:
1. Dilated CNN sees 63 timesteps with only 5 layers (vs 11 for regular CNN)
2. Exponential dilation = exponential receptive field growth
3. Causal padding ensures no future information leaks
4. Residual connections help gradient flow in deeper networks
5. Dilated CNN captures both short-term and long-term patterns efficiently
6. WaveNet (Google) and TCN use this exact architecture for time series
7. No recurrence = fully parallelizable = much faster training
""")
