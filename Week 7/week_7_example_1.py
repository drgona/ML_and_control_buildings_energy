import numpy as np
import matplotlib.pyplot as plt

# Define the quadratic objective function
def quadratic_objective(x):
    return x[0]**2 + x[1]**2  # Minimize x^2 + y^2

# Define the linear equality constraint: x + y - 1 = 0
def equality_constraint(x):
    return x[0] + x[1] - 1  # Reformulated as g(x) = 0

# Define the penalty function
def penalty_function(x, r):
    return quadratic_objective(x) + r * equality_constraint(x)**2

# Penalty Method Implementation
def penalty_method(initial_x, r_init=1, tol=1e-6, max_iter=50):
    x = np.array(initial_x, dtype=float)
    r = r_init  # Initial penalty parameter
    update_path = [x.copy()]  # Store iterations for visualization

    for _ in range(max_iter):
        # Compute gradients of the penalty function - analytic gradient
        grad_x = np.array([2*x[0], 2*x[1]]) + 2 * r * equality_constraint(x) * np.array([1, 1])

        # Perform gradient descent step
        step_size = 0.1 / r  # Decreasing step size as penalty increases
        # step_size = 0.0001
        x -= step_size * grad_x

        update_path.append(x.copy())

        # Increase the penalty parameter
        r *= 10

        # Check convergence (constraint satisfaction)
        if abs(equality_constraint(x)) < tol:
            break

    return x, update_path

# Solve using the Penalty Method
initial_x = [-0.50, 1.0]  # Initial guess
solution_x, update_path = penalty_method(initial_x)

# Convert update path to lists for visualization
update_x, update_y = zip(*update_path)

# Visualization of Penalty Method updates
fig, ax = plt.subplots(figsize=(8, 6))

# Create a grid for visualization
x_vals = np.linspace(-0.5, 1.5, 400)
y_vals = np.linspace(-0.5, 1.5, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = quadratic_objective([X, Y])

# Contour plot of the objective function
contour = ax.contour(X, Y, Z, levels=20, cmap="viridis")

# Plot the constraint line x + y = 1
constraint_x = np.linspace(-0.5, 1.5, 100)
constraint_y = 1 - constraint_x
ax.plot(constraint_x, constraint_y, 'r--', linewidth=2, label=r"Constraint: $x + y = 1$")

# Plot the Penalty Method update path
ax.plot(update_x, update_y, 'bo-', markersize=4, label="Penalty Method Updates")

# Mark the final solution
ax.plot(solution_x[0], solution_x[1], 'go', markersize=8, label="Optimal Solution")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Penalty Method for Quadratic Optimization with Linear Constraints")

ax.legend()
ax.grid(True)

plt.show(block=True)