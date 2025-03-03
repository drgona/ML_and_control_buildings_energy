import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define variables
x, y, λ = sp.symbols('x y λ')

# Define the objective function
f = x**2 + y**2  # Minimize x^2 + y^2

# Define the equality constraint
g = x + y - 1  # Constraint: x + y = 1

# Define the Lagrangian function
L = f + λ * g  # Lagrange function: L = f(x, y) + λ * g(x, y)

# Compute gradients (first-order conditions)
grad_L_x = sp.diff(L, x)  # ∂L/∂x
grad_L_y = sp.diff(L, y)  # ∂L/∂y
grad_L_λ = sp.diff(L, λ)  # ∂L/∂λ (enforces constraint)

# Solve the system of equations
solution = sp.solve([grad_L_x, grad_L_y, grad_L_λ], (x, y, λ))

# Extract solution
optimal_x, optimal_y, optimal_lambda = solution[x], solution[y], solution[λ]

# Display results
print(f"Optimal Solution: x = {optimal_x}, y = {optimal_y}")
print(f"Lagrange Multiplier: λ = {optimal_lambda}")

# Convert solutions to float for plotting
optimal_x = float(optimal_x)
optimal_y = float(optimal_y)

# Visualization of the Objective Function and Constraint
fig, ax = plt.subplots(figsize=(8, 6))

# Create a grid for visualization
x_vals = np.linspace(-0.5, 1.5, 400)
y_vals = np.linspace(-0.5, 1.5, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = X**2 + Y**2  # Objective function values

# Contour plot of the objective function
contour = ax.contour(X, Y, Z, levels=20, cmap="viridis")

# Plot the constraint line x + y = 1
constraint_x = np.linspace(-0.5, 1.5, 100)
constraint_y = 1 - constraint_x
ax.plot(constraint_x, constraint_y, 'r--', linewidth=2, label=r"Constraint: $x + y = 1$")

# Mark the optimal solution
ax.plot(optimal_x, optimal_y, 'go', markersize=8, label="Optimal Solution")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Lagrange Multiplier Method for Constrained Optimization")

ax.legend()
ax.grid(True)

plt.show(block=True)