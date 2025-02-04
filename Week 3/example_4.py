import torch
import matplotlib.pyplot as plt
import numpy as np

"""
Stability analysis of linear state space models 

autonomous systems setting (without inputs, i.e., no B matrix, only A matrix)

examples of:
1, stable system
2, unstable system
3, system with limit cycle

"""


"""
1, Stable system
"""

# Stable system (eigenvalues of A are negative)
A_stable = torch.tensor([[-1.0, -0.9],
                         [0.9, -0.5]], dtype=torch.float32)

# Initial condition
x0 = torch.tensor([1.0, 1.0], dtype=torch.float32)

# Simulation parameters
dt = 0.1
N = 200
t_values = np.arange(0, N*dt, dt)

# Simulate the system dynamics using Euler's method
x_values_stable = [x0]
for i in range(N):
    x_next = x_values_stable[-1] + dt * A_stable @ x_values_stable[-1]
    x_values_stable.append(x_next)

x_values_stable = torch.stack(x_values_stable).detach().numpy()

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(x_values_stable[:, 0], x_values_stable[:, 1], label="Stable System")
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Stable State-Space System Trajectory')
plt.grid()
plt.legend()
plt.show(block=True)


# Compute eigenvalues and eigenvectors using torch.linalg.eig()
L_complex, V_complex = torch.linalg.eig(A_stable)

# Display the results
print("Eigenvalues (complex form):")
print(L_complex)

# If you want the real and imaginary parts separately:
real_eigenvalues = L_complex.real
imaginary_eigenvalues = L_complex.imag

print("Eigenvalues (Real part):")
print(real_eigenvalues)

print("Eigenvalues (Imaginary part):")
print(imaginary_eigenvalues)



"""
2, Unstable system
"""

# Unstable system (eigenvalues of A are positive)
A_unstable = torch.tensor([[1.0, 0.0],
                           [0.0, 2.0]], dtype=torch.float32)

# Initial condition
x0 = torch.tensor([1.0, 1.0], dtype=torch.float32)

# Simulate the system dynamics using Euler's method
x_values_unstable = [x0]
for i in range(N):
    x_next = x_values_unstable[-1] + dt * A_unstable @ x_values_unstable[-1]
    x_values_unstable.append(x_next)

x_values_unstable = torch.stack(x_values_unstable).detach().numpy()

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(x_values_unstable[:, 0], x_values_unstable[:, 1], label="Unstable System")
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Unstable State-Space System Trajectory')
plt.grid()
plt.legend()
plt.show(block=True)

# Compute eigenvalues and eigenvectors using torch.linalg.eig()
L_complex, V_complex = torch.linalg.eig(A_unstable)

# Display the results
print("Eigenvalues (complex form):")
print(L_complex)

# If you want the real and imaginary parts separately:
real_eigenvalues = L_complex.real
imaginary_eigenvalues = L_complex.imag

print("Eigenvalues (Real part):")
print(real_eigenvalues)

print("Eigenvalues (Imaginary part):")
print(imaginary_eigenvalues)


"""
3,  Limit cycle system 
"""

# Limit cycle system (approximated by a linear system with complex eigenvalues)
A_limit_cycle = torch.tensor([[0.0, 1.0],
                              [-1.0, 0.0]], dtype=torch.float32)

# Initial condition (start from a point not at the origin)
x0 = torch.tensor([1.0, 0.0], dtype=torch.float32)

# Simulate the system dynamics using Euler's method
x_values_limit_cycle = [x0]
for i in range(N):
    x_next = x_values_limit_cycle[-1] + dt * A_limit_cycle @ x_values_limit_cycle[-1]
    x_values_limit_cycle.append(x_next)

x_values_limit_cycle = torch.stack(x_values_limit_cycle).detach().numpy()

# Plot the results
plt.figure(figsize=(10, 5))
plt.plot(x_values_limit_cycle[:, 0], x_values_limit_cycle[:, 1], label="Limit Cycle System")
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Limit Cycle State-Space System Trajectory')
plt.grid()
plt.legend()
plt.show(block=True)


# Compute eigenvalues and eigenvectors using torch.linalg.eig()
L_complex, V_complex = torch.linalg.eig(A_limit_cycle)

# Display the results
print("Eigenvalues (complex form):")
print(L_complex)

# If you want the real and imaginary parts separately:
real_eigenvalues = L_complex.real
imaginary_eigenvalues = L_complex.imag

print("Eigenvalues (Real part):")
print(real_eigenvalues)

print("Eigenvalues (Imaginary part):")
print(imaginary_eigenvalues)