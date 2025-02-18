import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)

# Generate synthetic regression data
X = torch.linspace(5, 10, 100).reshape(-1, 1)  # Range includes negative values to show ReLU effect
Y = 3 * X + 5 + torch.randn(X.shape) * 2  # Linear relation with noise

# Define Single-Layer Perceptron with ReLU Activation
class SingleLayerPerceptronReLU(nn.Module):
    def __init__(self):
        super(SingleLayerPerceptronReLU, self).__init__()
        self.linear = nn.Linear(1, 1)  # Single neuron (1 input → 1 output)
        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, x):
        return self.relu(self.linear(x))  # Apply ReLU after linear transformation

# Instantiate model
model = SingleLayerPerceptronReLU()

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for regression
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# Training loop
epochs = 500
for epoch in range(epochs):
    # Forward pass
    Y_pred = model(X)

    # Compute loss
    loss = criterion(Y_pred, Y)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 50 epochs
    if epoch % 50 == 0:
        print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}")

# Predictions after training
with torch.no_grad():
    Y_pred = model(X)

# Plot actual vs predicted values
plt.scatter(X.numpy(), Y.numpy(), label="Actual Data", color="red", alpha=0.5)
plt.plot(X.numpy(), Y_pred.numpy(), label="Fitted Line (ReLU)", color="blue", linewidth=2)
plt.xlabel("X (Input Feature)")
plt.ylabel("Y (Target Output)")
plt.title("Single-Layer Perceptron Regression with ReLU Activation")
plt.legend()
plt.show(block=True)
