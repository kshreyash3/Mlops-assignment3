# quantize.py

import joblib
import torch
import numpy as np
import os

# Load sklearn model
sk_model = joblib.load("models/sklearn_model.joblib")

# Extract weights and bias
weights = sk_model.coef_
bias = sk_model.intercept_

# Save original (float) params
unquant_params = {
    "weights": weights,
    "bias": bias
}
os.makedirs("quantized", exist_ok=True)
joblib.dump(unquant_params, "quantized/unquant_params.joblib")

# Manual quantization to uint8
w_min, w_max = weights.min(), weights.max()
scale = (w_max - w_min) / 255
zero_point = np.round(-w_min / scale).astype(np.uint8)

weights_q = np.round((weights - w_min) / scale).astype(np.uint8)
bias_q = np.round(bias).astype(np.int32)

quant_params = {
    "weights_q": weights_q,
    "bias_q": bias_q,
    "scale": scale,
    "zero_point": zero_point
}
joblib.dump(quant_params, "quantized/quant_params.joblib")

# Reconstruct PyTorch model and dequantize for inference
class SimpleModel(torch.nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)

model = SimpleModel(in_features=len(weights))

# Dequantize weights
w_dequant = (weights_q.astype(np.float32) * scale) + w_min
b_dequant = bias_q.astype(np.float32)

# Load into PyTorch model
model.linear.weight.data = torch.tensor([w_dequant], dtype=torch.float32)
model.linear.bias.data = torch.tensor([b_dequant], dtype=torch.float32)

# Run inference
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X, y = fetch_california_housing(return_X_y=True)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_pred = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy().flatten()
r2 = r2_score(y_test, y_pred)
print(f"[Quantized] R² Score: {r2:.4f}")
