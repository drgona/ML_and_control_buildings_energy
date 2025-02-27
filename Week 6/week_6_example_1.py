import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

"""
Signal processing for system identification

"""

# Parameters
g = 9.81  # Gravitational acceleration (m/s^2)
l = 1.0   # Length of pendulum (m)
b = 0.5   # Damping coefficient
dt = 0.01 # Time step
noise_std = 0.2  # Standard deviation of noise

# Low-pass filter parameters
cutoff_freq = 2.0  # Cutoff frequency in Hz
sampling_rate = 1 / dt  # Sampling rate in Hz
order = 2  # Filter order

# Create Butterworth low-pass filter
def butter_lowpass_filter(data, cutoff, fs, order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Normalize function
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Dynamics function
def pendulum_dynamics(state):
    theta, omega = state
    dtheta = omega
    domega = -b * omega - (g / l) * torch.sin(theta)
    return torch.tensor([dtheta, domega])

# Simulate the pendulum
def simulate_pendulum(theta0, omega0, steps):
    state = torch.tensor([theta0, omega0], dtype=torch.float32)
    trajectory = [state.clone()]
    for _ in range(steps):
        state += pendulum_dynamics(state) * dt
        trajectory.append(state.clone())
    return torch.stack(trajectory)

# Initial conditions
theta0 = 1.0  # Initial angle (radians)
omega0 = 0.0  # Initial angular velocity (rad/s)
steps = 1000

# Simulate and visualize
trajectory = simulate_pendulum(theta0, omega0, steps)

# Extract states and add noise
theta_noisy = trajectory[:, 0].numpy() + np.random.normal(0, noise_std, size=len(trajectory))
omega_noisy = trajectory[:, 1].numpy() + np.random.normal(0, noise_std, size=len(trajectory))
time = np.arange(len(theta_noisy)) * dt


"""
Filter
"""

# Apply low-pass filter
theta_filtered = butter_lowpass_filter(theta_noisy, cutoff_freq, sampling_rate, order)
omega_filtered = butter_lowpass_filter(omega_noisy, cutoff_freq, sampling_rate, order)

# Time series with noise and filtered output
plt.figure(figsize=(6, 6))

plt.subplot(2, 1, 1)
plt.plot(time, theta_noisy, label="Noisy Theta (Angle)", alpha=0.5)
plt.plot(time, theta_filtered, label="Filtered Theta (Angle)", linewidth=2)
plt.ylabel("Theta")
plt.title("Low-Pass Filtered Time Series")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time, omega_noisy, label="Noisy Omega (Angular Velocity)", alpha=0.5)
plt.plot(time, omega_filtered, label="Filtered Omega (Angular Velocity)", linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Omega")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show(block=True)


"""
Normalize
"""

# Normalize filtered data
theta_filtered_norm = normalize(theta_filtered)
omega_filtered_norm = normalize(omega_filtered)

# Normalized Time series with noise and filtered output
plt.figure(figsize=(6, 6))

plt.subplot(2, 1, 1)
plt.plot(time, theta_filtered, label="Filtered Theta (Angle)", alpha=0.5, linewidth=2)
plt.plot(time, theta_filtered_norm, label="Filtered & Normalized Theta (Angle)", linewidth=2)
plt.ylabel("Theta (normalized)")
plt.title("Normalized Time Series")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time, omega_filtered, label="Filtered Omega (Angular Velocity)", alpha=0.5, linewidth=2)
plt.plot(time, omega_filtered_norm, label="Filtered & Normalized Omega (Angular Velocity)", linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Omega (normalized)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show(block=True)

