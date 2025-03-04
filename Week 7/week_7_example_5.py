import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

"""
Solve quadratic program using CVXPY
"""

# Define problem data
Q = np.array([[4, 1], [1, 2]])  # Quadratic cost matrix
c = np.array([1, 1])            # Linear cost vector
A = np.array([[1, 1], [-1, 2], [2, 1]])  # Constraint coefficients
b = np.array([2, 2, 3])        # Constraint bounds

# Define variables
x = cp.Variable(2)

# Define quadratic objective function
objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + c @ x)

# Define constraints
constraints = [A @ x <= b]

# Solve the problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Extract solution
optimal_x = x.value

# Print the optimal solution
print("Optimal x:", optimal_x)
print("Optimal cost:", problem.value)

# Visualization of feasible region and optimal point
x_vals = np.linspace(-1, 3, 400)
y_vals = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = 0.5 * (Q[0, 0] * X**2 + 2 * Q[0, 1] * X * Y + Q[1, 1] * Y**2) + c[0] * X + c[1] * Y

# Plot the contour of the quadratic function
plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=20, cmap='coolwarm')

# Define feasible region
feasible_x = []
feasible_y = []
for x1 in x_vals:
    for x2 in y_vals:
        if np.all(A @ np.array([x1, x2]) <= b):
            feasible_x.append(x1)
            feasible_y.append(x2)
plt.scatter(feasible_x, feasible_y, s=1, color='lightgray', alpha=0.5)

# Plot constraints as lines
for i in range(A.shape[0]):
    a1, a2 = A[i]
    if a2 != 0:
        y_line = (b[i] - a1 * x_vals) / a2
        plt.plot(x_vals, y_line, 'k', lw=1)
    else:
        plt.axvline(b[i] / a1, color='k', lw=1)

# Plot the optimal point
plt.scatter(optimal_x[0], optimal_x[1], color='red', marker='o', s=100, label='Optimal Solution')
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.title("Quadratic Program Visualization")
plt.xlim([-1, 3])
plt.ylim([-1, 3])
plt.grid()
plt.show(block=True)