import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


"""
example of auoregressive with exogenous input (ARX) model

"""


# Generate synthetic data for ARX model
def generate_arx_data(steps=1000):
    u = torch.sin(torch.linspace(0, 10, steps)).reshape(-1, 1)  # Sine wave input
    y = torch.zeros(steps, 1)  # Output

    # Define ARX system: y(t) = 0.7*y(t-1) + 0.2*u(t-1)
    for t in range(1, steps):
        y[t] = 0.7 * y[t - 1] + 0.2 * u[t - 1] + 0.1 * torch.randn(1)  # Adding noise

    return u, y


# Define a linear ARX model using PyTorch
class LinearARXModel(nn.Module):
    def __init__(self):
        super(LinearARXModel, self).__init__()
        self.linear = nn.Linear(2, 1)  # Two inputs: y(t-1), u(t-1)

    def forward(self, x):
        return self.linear(x)


# Generate data
u, y = generate_arx_data()

# Prepare training data
X_train = torch.cat((y[:-1], u[:-1]), dim=1)
y_train = y[1:]

# Initialize and train the model
model = LinearARXModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(500):
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

# Predict and plot
with torch.no_grad():
    y_pred = model(X_train)

plt.figure(figsize=(6, 4))
plt.plot(y.numpy(), label='True Output', alpha=0.6)
plt.plot(y_pred.numpy(), label='Predicted Output', linestyle='dashed')
plt.xlabel('Time Step')
plt.ylabel('Output')
plt.legend()
plt.title('Linear ARX Model Prediction')
plt.grid()
plt.show(block=True)