"""
01. OLS vs Deep Learning - The Simplest Comparison
=====================================================
Goal: Show that a simple neural network with 1 layer and linear activation
      is mathematically equivalent to OLS regression.
      Then show how adding layers/nonlinearity improves prediction.

Key Takeaway:
  - OLS: y = X @ beta  (closed-form solution)
  - Neural Net (1 layer, linear): y = X @ W + b  (same thing, learned via gradient descent)
  - Neural Net (deep, nonlinear): y = f(f(f(X)))  (can capture nonlinear patterns)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# 1. Download Data
# ============================================================
print("=" * 60)
print("Step 1: Downloading SPY data from Yahoo Finance")
print("=" * 60)

df = yf.download("SPY", start="2015-01-01", end="2025-12-31", auto_adjust=True)
df = df[["Close"]].copy()
df.columns = ["Close"]
df["Return"] = df["Close"].pct_change()
df.dropna(inplace=True)

# Create features: lagged returns (past 5 days predict next day return)
for i in range(1, 6):
    df[f"Lag_{i}"] = df["Return"].shift(i)
df.dropna(inplace=True)

# Target: next day return
df["Target"] = df["Return"].shift(-1)
df.dropna(inplace=True)

feature_cols = [f"Lag_{i}" for i in range(1, 6)]
X = df[feature_cols].values
y = df["Target"].values.reshape(-1, 1)

# Train/Test split (80/20, no shuffle for time series)
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Standardize
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_s = scaler_X.fit_transform(X_train)
X_test_s = scaler_X.transform(X_test)
y_train_s = scaler_y.fit_transform(y_train)
y_test_s = scaler_y.transform(y_test)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
print(f"Features: {feature_cols}")

# ============================================================
# 2. OLS Regression
# ============================================================
print("\n" + "=" * 60)
print("Step 2: OLS Regression (sklearn)")
print("=" * 60)

ols = LinearRegression()
ols.fit(X_train_s, y_train_s)
y_pred_ols = ols.predict(X_test_s)

# Inverse transform
y_pred_ols_orig = scaler_y.inverse_transform(y_pred_ols)
mse_ols = mean_squared_error(y_test, y_pred_ols_orig)
r2_ols = r2_score(y_test, y_pred_ols_orig)

print(f"OLS Coefficients: {ols.coef_.flatten()}")
print(f"OLS Intercept:    {ols.intercept_[0]:.6f}")
print(f"OLS MSE:          {mse_ols:.8f}")
print(f"OLS R2:           {r2_ols:.6f}")

# ============================================================
# 3. Neural Net - Linear (equivalent to OLS)
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Neural Network - LINEAR (should match OLS)")
print("=" * 60)


class LinearNet(nn.Module):
    """Single layer, no activation = same as OLS"""
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)


# Convert to tensors
X_train_t = torch.FloatTensor(X_train_s)
y_train_t = torch.FloatTensor(y_train_s)
X_test_t = torch.FloatTensor(X_test_s)

# Train
model_linear = LinearNet(5)
optimizer = torch.optim.Adam(model_linear.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(500):
    model_linear.train()
    pred = model_linear(X_train_t)
    loss = criterion(pred, y_train_t)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model_linear.eval()
with torch.no_grad():
    y_pred_lin = model_linear(X_test_t).numpy()

y_pred_lin_orig = scaler_y.inverse_transform(y_pred_lin)
mse_lin = mean_squared_error(y_test, y_pred_lin_orig)
r2_lin = r2_score(y_test, y_pred_lin_orig)

# Show weights (should be close to OLS coefficients)
w = model_linear.linear.weight.data.numpy().flatten()
b = model_linear.linear.bias.data.numpy()[0]
print(f"NN-Linear Weights:   {w}")
print(f"NN-Linear Bias:      {b:.6f}")
print(f"NN-Linear MSE:       {mse_lin:.8f}")
print(f"NN-Linear R2:        {r2_lin:.6f}")
print(f"\nCompare with OLS:    {ols.coef_.flatten()}")
print(">>> Weights should be very similar! <<<")

# ============================================================
# 4. Neural Net - Deep (nonlinear, can capture more patterns)
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Neural Network - DEEP (nonlinear)")
print("=" * 60)


class DeepNet(nn.Module):
    """Multi-layer with ReLU activations"""
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)


dataset = TensorDataset(X_train_t, y_train_t)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model_deep = DeepNet(5)
optimizer = torch.optim.Adam(model_deep.parameters(), lr=0.001)
criterion = nn.MSELoss()

losses = []
for epoch in range(200):
    model_deep.train()
    epoch_loss = 0
    for xb, yb in loader:
        pred = model_deep(xb)
        loss = criterion(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    losses.append(epoch_loss / len(loader))
    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1}/200, Loss: {losses[-1]:.6f}")

model_deep.eval()
with torch.no_grad():
    y_pred_deep = model_deep(X_test_t).numpy()

y_pred_deep_orig = scaler_y.inverse_transform(y_pred_deep)
mse_deep = mean_squared_error(y_test, y_pred_deep_orig)
r2_deep = r2_score(y_test, y_pred_deep_orig)

print(f"Deep NN MSE:  {mse_deep:.8f}")
print(f"Deep NN R2:   {r2_deep:.6f}")

# ============================================================
# 5. Plot Comparison
# ============================================================
print("\n" + "=" * 60)
print("Step 5: Plotting results")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("OLS vs Deep Learning - Financial Return Prediction", fontsize=14, fontweight="bold")

# (a) Training loss curve
axes[0, 0].plot(losses, color="#3b82f6", linewidth=1.5)
axes[0, 0].set_title("Deep NN Training Loss")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("MSE Loss")
axes[0, 0].grid(True, alpha=0.3)

# (b) Actual vs Predicted (last 100 points)
n_show = 100
axes[0, 1].plot(y_test[-n_show:], label="Actual", color="#64748b", linewidth=1)
axes[0, 1].plot(y_pred_ols_orig[-n_show:], label="OLS", color="#ef4444", linewidth=1, alpha=0.8)
axes[0, 1].plot(y_pred_deep_orig[-n_show:], label="Deep NN", color="#3b82f6", linewidth=1, alpha=0.8)
axes[0, 1].set_title("Actual vs Predicted Returns (last 100 days)")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# (c) Scatter: OLS
axes[1, 0].scatter(y_test, y_pred_ols_orig, alpha=0.3, s=10, color="#ef4444")
axes[1, 0].plot([-0.05, 0.05], [-0.05, 0.05], "k--", linewidth=1)
axes[1, 0].set_title(f"OLS: R2={r2_ols:.4f}")
axes[1, 0].set_xlabel("Actual")
axes[1, 0].set_ylabel("Predicted")
axes[1, 0].grid(True, alpha=0.3)

# (d) Scatter: Deep NN
axes[1, 1].scatter(y_test, y_pred_deep_orig, alpha=0.3, s=10, color="#3b82f6")
axes[1, 1].plot([-0.05, 0.05], [-0.05, 0.05], "k--", linewidth=1)
axes[1, 1].set_title(f"Deep NN: R2={r2_deep:.4f}")
axes[1, 1].set_xlabel("Actual")
axes[1, 1].set_ylabel("Predicted")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/mnt/nas/Class/2026/deeplearning/01_ols_vs_deeplearning.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: 01_ols_vs_deeplearning.png")

# ============================================================
# 6. Summary Table
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
summary = pd.DataFrame({
    "Model": ["OLS", "NN-Linear (1 layer)", "NN-Deep (3 layers)"],
    "MSE": [mse_ols, mse_lin, mse_deep],
    "R2": [r2_ols, r2_lin, r2_deep],
    "Learnable Params": [
        6,  # 5 weights + 1 bias
        6,  # same
        sum(p.numel() for p in model_deep.parameters()),
    ],
})
print(summary.to_string(index=False))
print("""
KEY INSIGHTS:
1. NN-Linear ~ OLS: Same weights, same performance (both are linear models)
2. Deep NN has more capacity but may not always beat OLS on noisy financial data
3. Financial returns are notoriously hard to predict (low R2 is normal)
4. Deep learning shines when there are nonlinear patterns in the data
""")
