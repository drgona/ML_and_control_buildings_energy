import torch

"""
automatic differentiation example in PyTorch
"""

x = torch.tensor(3.0, requires_grad=True)
z = torch.tensor(2.0, requires_grad=True)

# Define function f(x, z) = x^2 * z + sin(x)
f = x**2 * z + torch.sin(x)

# Compute gradients
f.backward()

# Print partial derivatives
print(f"∂f/∂x at (x={x.item()}, z={z.item()}): {x.grad.item()}")
print(f"∂f/∂z at (x={x.item()}, z={z.item()}): {z.grad.item()}")
