
import torch
import numpy as np
import pandas as pd
import sklearn
import matplotlib

# Create a sample satellite time-series tensor
x = torch.randn(4, 10, 3)

print("PyTorch version:", torch.__version__)
print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("Tensor shape:", x.shape)
print("GPU available:", torch.cuda.is_available())
print("Setup successful!")
