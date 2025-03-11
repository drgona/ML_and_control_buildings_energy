import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np

# Create CartPole environment
env = gym.make("CartPole-v1")

# Define RL model using PPO
model = PPO("MlpPolicy", env, verbose=0)

# Train the model
TIMESTEPS = 10000
model.learn(total_timesteps=TIMESTEPS)

# Evaluate the trained model
def evaluate_model(env, model, episodes=10):
    rewards = []
    for _ in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
        rewards.append(episode_reward)
    return rewards

# Simulate trained policy and collect state-action data
def simulate_policy(env, model):
    obs, _ = env.reset()
    done = False
    states = []
    actions = []
    time_steps = []
    t = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        states.append(obs)
        actions.append(action)
        time_steps.append(t)
        obs, _, done, _, _ = env.step(action)
        t += 1
    return np.array(time_steps), np.array(states), np.array(actions)

# Run evaluation
episode_rewards = evaluate_model(env, model)

# Simulate control policy
time_steps, states, actions = simulate_policy(env, model)

# Plot total rewards over episodes
plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(episode_rewards)), episode_rewards, marker='o', linestyle='-')
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Performance of PPO on CartPole-v1")
plt.grid()
plt.show(block=True)

# Plot state variables over time
state_labels = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]
plt.figure(figsize=(10, 6))
for i in range(states.shape[1]):
    plt.plot(time_steps, states[:, i], label=state_labels[i])
plt.xlabel("Time Step")
plt.ylabel("State Values")
plt.title("CartPole State Evolution Over Time")
plt.legend()
plt.grid()
plt.show(block=True)

# Plot actions over time
plt.figure(figsize=(10, 4))
plt.step(time_steps, actions, where='mid', label="Actions")
plt.xlabel("Time Step")
plt.ylabel("Action (0=Left, 1=Right)")
plt.title("CartPole Actions Over Time")
plt.legend()
plt.grid()
plt.show(block=True)

# Close the environment
env.close()