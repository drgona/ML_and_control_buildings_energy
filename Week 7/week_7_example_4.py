import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

# Define problem data
c = np.array([-3, -2])  # Coefficients for the objective function (maximize 3x + 2y)
A = np.array([[1, 1], [1, -1], [-1, 1]])  # Modified constraint coefficients
b = np.array([4, 1, 1])  # Modified constraint bounds to ensure active constraint

# Define variables
x = cp.Variable(2)

# Define objective function (maximization -> minimize negative)
objective = cp.Minimize(c @ x)

# Define constraints
constraints = [A @ x <= b, x >= 0]

# Solve the problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Extract solution
optimal_x = x.value

# Print the optimal solution
print("Optimal x:", optimal_x)
print("Optimal cost:", problem.value)

# Visualization of feasible region and optimal point
x_vals = np.linspace(0, 5, 400)
y_vals = np.linspace(0, 5, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = c[0] * X + c[1] * Y

# Plot feasible region
plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=20, cmap='coolwarm')

feasible_x = []
feasible_y = []
for x1 in x_vals:
    for x2 in y_vals:
        if np.all(A @ np.array([x1, x2]) <= b) and x1 >= 0 and x2 >= 0:
            feasible_x.append(x1)
            feasible_y.append(x2)
plt.scatter(feasible_x, feasible_y, s=1, color='lightgray', alpha=0.5)

# Plot constraint lines
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
plt.title("Linear Program Visualization with Active Constraint")
plt.xlim([0, 5])
plt.ylim([0, 5])
plt.grid()
plt.show(block=True)