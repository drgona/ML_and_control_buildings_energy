import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# Create an Opti instance
opti = ca.Opti()

# Define optimization variables
x = opti.variable()
y = opti.variable()

# Define objective function (nonlinear)
opti.minimize((x - 2)**2 + (y - 3)**2)

# Define nonlinear constraints
opti.subject_to(x**2 + y <= 5)  # x^2 + y <= 5
opti.subject_to(x + y**2 >= 3)  # x + y^2 >= 3

# Bounds on variables
opti.subject_to(0 <= x)
opti.subject_to(0 <= y)

# Solver options
opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.tol': 1e-6}
opti.solver('ipopt', opts)

# Solve the problem
opti.set_initial(x, 2)
opti.set_initial(y, 2)
sol = opti.solve()

# Extract results
optimal_x = sol.value(x)
optimal_y = sol.value(y)

# Print results
print("Optimal x:", optimal_x)
print("Optimal y:", optimal_y)

# Visualization
x_vals = np.linspace(0, 5, 400)
y_vals = np.linspace(0, 5, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z = (X - 2)**2 + (Y - 3)**2

plt.figure(figsize=(8, 6))
plt.contour(X, Y, Z, levels=20, cmap='coolwarm')

# Plot constraints
x_line = np.linspace(0, 5, 100)
y_con1 = 5 - x_line**2  # Reformulated constraint1 as y <= 5 - x^2
y_con2 = np.sqrt(3 - x_line)  # Reformulated constraint2 as y >= sqrt(3 - x)
plt.plot(x_line, y_con1, 'k', lw=2, label='x^2 + y <= 5')
plt.plot(x_line, y_con2, 'b', lw=2, label='x + y^2 >= 3')
plt.fill_between(x_line, np.maximum(0, y_con2), np.minimum(5, y_con1), color='lightgray', alpha=0.5)

# Plot optimal solution
plt.scatter(optimal_x, optimal_y, color='red', marker='o', s=100, label='Optimal Solution')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Nonlinear Programming Visualization with CasADi Opti and IPOPT")
plt.xlim([0, 5])
plt.ylim([0, 5])
plt.grid()
plt.show(block=True)
