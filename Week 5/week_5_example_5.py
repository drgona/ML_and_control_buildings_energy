import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
torch.manual_seed(42)

# Generate synthetic electricity load data (Time Series)
time = np.arange(0, 100, 0.1)  # 1000 time points
load = 50 + 10 * np.sin(0.1 * time) + 5 * np.random.randn(len(time))  # Sinusoidal pattern with noise

# Prepare dataset: Using past 10 time steps to predict the next load
seq_length = 10
X, Y = [], []
for i in range(len(load) - seq_length):
    X.append(load[i:i + seq_length])  # Input: past 10 values
    Y.append(load[i + seq_length])    # Output: next value

X, Y = np.array(X), np.array(Y).reshape(-1, 1)  # Ensure Y is 2D before scaling

# Normalize data
scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()
X = scaler_X.fit_transform(X)
Y = scaler_Y.fit_transform(Y)  # Now correctly shaped

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train, Y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32)
X_test, Y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32)

# Define MLP model
class MLPForecast(nn.Module):
    def __init__(self, input_size):
        super(MLPForecast, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)  # Fully connected layer 1
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)  # Fully connected layer 2
        self.fc3 = nn.Linear(32, 1)   # Output layer

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)  # Linear output for regression

# Instantiate model
model = MLPForecast(input_size=seq_length)

# Loss function & optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 500
for epoch in range(epochs):
    # Forward pass
    Y_pred = model(X_train)

    # Compute loss
    loss = criterion(Y_pred, Y_train)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 50 epochs
    if epoch % 50 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

# Make predictions
with torch.no_grad():
    Y_pred = model(X_test)

# Convert predictions back to original scale
Y_pred = scaler_Y.inverse_transform(Y_pred.numpy())
Y_test = scaler_Y.inverse_transform(Y_test.numpy())

# Plot actual vs predicted load
plt.figure(figsize=(10, 5))
plt.plot(Y_test, label="Actual Load", color="red")
plt.plot(Y_pred, label="Predicted Load", color="blue", linestyle="dashed")
plt.xlabel("Time")
plt.ylabel("Load")
plt.title("MLP Load Forecasting (PyTorch)")
plt.legend()
plt.show(block=True)
