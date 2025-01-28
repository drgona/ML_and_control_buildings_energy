import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data for a nonlinear function
def nonlinear_function(x):
    return np.sin(2 * np.pi * x) + 0.1 * np.random.randn(*x.shape)  # Sine wave with noise

# Create input and output data
np.random.seed(42)
x = np.linspace(0, 1, 100).reshape(-1, 1)  # Inputs
y = nonlinear_function(x)                  # Outputs

# Convert data to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Define a simple regression model
class SimpleRegressor(nn.Module):
    def __init__(self):
        super(SimpleRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

# Instantiate the model, define loss and optimizer
model = SimpleRegressor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 1000
losses = []
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    predictions = model(x_tensor)
    loss = criterion(predictions, y_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()
plt.show(block=True)

# Evaluate the model and plot results
model.eval()
with torch.no_grad():
    y_pred = model(x_tensor).numpy()

# Plot the original data and the model's predictions
plt.figure(figsize=(10, 5))
plt.scatter(x, y, label='True Data', color='blue')
plt.plot(x, y_pred, label='Model Prediction', color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Nonlinear Function Regression')
plt.legend()
plt.show()
plt.show(block=True)

