import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

"""
Example 4

PID control of a simple mass-spring-damper system


"""


# Define the system parameters
m = 1.0  # mass (kg)
b = 0.5  # damping coefficient (N·s/m)
k = 2.0  # spring constant (N/m)

# Define the PID controller parameters
Kp = 200.0  # Proportional gain
Ki = 0.1  # Integral gain
Kd = 20.0  # Derivative gain

# Define the desired position (setpoint)
x_setpoint = 1.0  # desired position (m)

# PID control variables
integral = 0
previous_error = 0


# Define the system of ODEs
def system(t, y, m, b, k, F):
    # y[0] is position, y[1] is velocity
    dxdt = y[1]
    dvdt = (F - b * y[1] - k * y[0]) / m
    return [dxdt, dvdt]


# PID controller to compute force
def pid_controller(error, dt):
    global integral, previous_error
    integral += error * dt
    derivative = 0 if dt == 0 else (error - previous_error) / dt  # Avoid divide by zero
    previous_error = error
    return Kp * error + Ki * integral + Kd * derivative


# Define the simulation function
def simulate_pid_control(t_span, initial_conditions):
    global integral, previous_error
    t_values = []
    x_values = []
    v_values = []

    def control_system(t, y):
        error = x_setpoint - y[0]
        dt = t - t_values[-1] if t_values else 0  # Time step (ensure it's not zero)
        F = pid_controller(error, dt)  # Compute force using PID
        t_values.append(t)
        x_values.append(y[0])
        v_values.append(y[1])
        return system(t, y, m, b, k, F)

    # Solve the ODE
    sol = solve_ivp(control_system, t_span, initial_conditions, t_eval=np.linspace(t_span[0], t_span[1], 1000))

    # Check if the solution is structured correctly and return position and velocity
    return sol.t, sol.y[0], sol.y[1]  # Return time, position, and velocity


# Initial conditions: initial position and velocity
initial_conditions = [0, 0]  # starting at position 0 and velocity 0

# Time span for the simulation
t_span = (0, 5)

# Simulate the system
t, x, v = simulate_pid_control(t_span, initial_conditions)

# Plot the results
plt.figure(figsize=(12, 6))

# Plot position over time
plt.subplot(2, 1, 1)
plt.plot(t, x, label='Position (x)', color='b')
plt.axhline(x_setpoint, color='r', linestyle='--', label='Setpoint')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.title('PID Control: Position vs Time')
plt.legend()

# Plot velocity over time
plt.subplot(2, 1, 2)
plt.plot(t, v, label='Velocity (v)', color='g')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('PID Control: Velocity vs Time')
plt.legend()

plt.tight_layout()
plt.show(block=True)
