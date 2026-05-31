
import torch
import numpy as np

predictions = torch.tensor([
    [[0.1, 0.2, 0.7], [0.8, 0.1, 0.1]],
    [[0.3, 0.4, 0.3], [0.1, 0.1, 0.8]]
])
targets = torch.tensor([
    [2, 0],
    [1, 2]
], dtype=torch.int64)

print(f"Predictions shape: {predictions.shape}")
print(f"Targets shape: {targets.shape}")

targets_expanded = targets[:, None]
print(f"Targets expanded shape: {targets_expanded.shape}")

try:
    targets_values = torch.take_along_dim(predictions, targets_expanded, dim=-1)
    print(f"Targets values shape: {targets_values.shape}")
except Exception as e:
    print(f"take_along_dim failed: {e}")

k = 1
topk_values, _ = torch.topk(predictions, k, sorted=True)
print(f"Topk values shape: {topk_values.shape}")
