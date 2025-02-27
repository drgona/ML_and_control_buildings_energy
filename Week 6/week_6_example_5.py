import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

"""
Neural State-Space Model (NSSM) for System Identification

"""

# Define Neural State-Space Model
class NeuralStateSpaceModel(nn.Module):
    def __init__(self, state_dim, input_dim, hidden_dim=64):
        super(NeuralStateSpaceModel, self).__init__()
        # state equation
        self.f = nn.Sequential(
            nn.Linear(state_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        # output equation
        self.g = nn.Sequential(
            nn.Linear(state_dim + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, u):
        xu = torch.cat((x, u), dim=1)  # Concatenate state and input
        x_next = x + self.f(xu)  # State update equation
        y_pred = self.g(xu)  # Output equation
        return x_next, y_pred


# Generate synthetic training data of a linear system
def generate_data(steps=1000):
    u = torch.sin(torch.linspace(0, 10, steps)).reshape(-1, 1)  # Sine wave input
    x = torch.zeros(steps, 2)  # State (e.g., position and velocity)
    y = torch.zeros(steps, 1)  # Output

    for t in range(1, steps):
        x[t, 0] = x[t - 1, 0] + x[t - 1, 1] * 0.01 + 0.5 * u[t - 1, 0] * 0.01  # Position update
        x[t, 1] = x[t - 1, 1] - 0.2 * x[t - 1, 0] * 0.01  # Velocity update
        y[t] = x[t, 0]  # Output is position

    return x, u, y


# Training the NSSM
def train_nssm(model, x, u, y, epochs=500, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        x_pred, y_pred = model(x[:-1], u[:-1])
        loss = loss_fn(y_pred, y[1:])
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    return model


# Generate data
x, u, y = generate_data()

# Initialize and train the model
state_dim = 2
input_dim = 1
nssm = NeuralStateSpaceModel(state_dim, input_dim)
nssm = train_nssm(nssm, x, u, y)

# Test the trained model
x_pred, y_pred = nssm(x[:-1], u[:-1])

# Plot results
plt.figure(figsize=(6, 4))
plt.plot(y.numpy(), label='True Output', alpha=0.6)
plt.plot(y_pred.detach().numpy(), label='Predicted Output', linestyle='dashed')
plt.xlabel('Time Step')
plt.ylabel('Output')
plt.legend()
plt.title('Neural State-Space Model Prediction')
plt.grid()
plt.show(block=True)

