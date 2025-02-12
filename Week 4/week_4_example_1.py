import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

"""
Example 1 

Role of feedback in closed-loop dynamical systems

system: 1D linear ordinary differential equation 
controller: linear state feedback

"""


# Define the system with feedback
def system(x, t, a, b, k, feedback_type):
    if feedback_type == 'positive':
        u = k * x  # Positive feedback
    elif feedback_type == 'negative':
        u = -k * x  # Negative feedback
    else:
        u = 0

    dxdt = -a * x + b * u  # Dynamics of the system
    return dxdt


# Parameters
a = 0.5  # Dissipation constant (decay if a > 0)
b = 1.0  # Control influence on the system
k = 2.0 # feedback gain

# Initial condition
x0 = 1.0  # Initial state

# Time vector
t = np.linspace(0, 10, 1000)

# Solve the system for positive feedback
x_positive = odeint(system, x0, t, args=(a, b, k, 'positive'))

# Solve the system for negative feedback
x_negative = odeint(system, x0, t, args=(a, b, k, 'negative'))

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(t, x_negative, label="Negative Feedback (k=2)", color='r')
plt.title('1D Dynamical System with Negative Feedback')
plt.xlabel('Time (t)')
plt.ylabel('State x(t)')
plt.legend()
plt.grid(True)
plt.show(block=True)

plt.figure(figsize=(10, 6))
plt.plot(t, x_positive, label="Positive Feedback (k=2)", color='b')
plt.title('1D Dynamical System with Positive Feedback')
plt.xlabel('Time (t)')
plt.ylabel('State x(t)')
plt.legend()
plt.grid(True)
plt.show(block=True)

