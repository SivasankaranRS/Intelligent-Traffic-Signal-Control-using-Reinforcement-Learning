import os
import sys

import gymnasium as gym
from stable_baselines3.dqn.dqn import DQN

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci

from sumo_rl import SumoEnvironment

# Monkey patch to fix the AttributeError in close
LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ
def patched_close(self):
    if not hasattr(self, 'sumo') or self.sumo is None:
        return
    if not LIBSUMO:
        traci.switch(self.label)
    traci.close()
    if self.disp is not None:
        self.disp.stop()
        self.disp = None

SumoEnvironment.close = patched_close

# Monkey patch to fix the AttributeError in close
LIBSUMO = "LIBSUMO_AS_TRACI" in os.environ
def patched_close(self):
    if not hasattr(self, 'sumo') or self.sumo is None:
        return
    if not LIBSUMO:
        traci.switch(self.label)
    traci.close()
    if self.disp is not None:
        self.disp.stop()
        self.disp = None

SumoEnvironment.close = patched_close


env = SumoEnvironment(
    net_file=r"..\..\..\dqn\env_iiser\map.net.xml",
    route_file=r"..\..\..\dqn\env_iiser\routes_new.rou.xml",
    out_csv_name=r"dqnoutput",
    single_agent=True,
    use_gui=False,
    num_seconds=5000,
)

model = DQN(
        env=env,
        policy="MlpPolicy",
        learning_rate=0.00001,
        learning_starts=0,
        train_freq=1,
        target_update_interval=1000,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        exploration_fraction=0.5,
        buffer_size=500000,
        verbose=1,
        tensorboard_log="./dqn_tensorboard/",
        seed=42,
    )

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ProgressBarCallback, EvalCallback

class RewardLoggingCallback(BaseCallback):
    """Callback for logging and tracking episode rewards during training."""
    def __init__(self):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        # Track done episodes from dones array
        if len(self.model.env.buf_dones) > 0:
            if self.model.env.buf_dones[0]:  # Episode finished
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)
                self.episode_count += 1
                self.current_episode_reward = 0
                self.current_episode_length = 0
        
        # Track current step reward
        if len(self.model.env.buf_rews) > 0:
            self.current_episode_reward += self.model.env.buf_rews[0]
        self.current_episode_length += 1
        
        return True

# Create callback instances
reward_callback = RewardLoggingCallback()
progress_callback = ProgressBarCallback()

# Combine callbacks (removed eval_callback for faster training)
callbacks = CallbackList([reward_callback, progress_callback])

try:
    model.learn(total_timesteps=100000, log_interval=1000, callback=callbacks)
except Exception as e:
    print(f"Training failed: {e}")
    # Optionally save partial model
    model.save("dqn_iiser_partial_model")
    print("Partial model saved.")
finally:
    env.close()

# Plot Reward Convergence
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Episode Rewards
axes[0].plot(reward_callback.episode_rewards, linewidth=1.5, alpha=0.7, label='Episode Reward')
if len(reward_callback.episode_rewards) > 0:
    # Add moving average
    window = min(100, max(1, len(reward_callback.episode_rewards) // 10))
    moving_avg = np.convolve(reward_callback.episode_rewards, np.ones(window)/window, mode='valid')
    axes[0].plot(range(window-1, window-1+len(moving_avg)), moving_avg, linewidth=2.5, 
                 color='red', label=f'Moving Average (window={window})')

axes[0].set_xlabel('Episode', fontsize=11)
axes[0].set_ylabel('Reward', fontsize=11)
axes[0].set_title('DQN Training: Episode Rewards Over Time', fontsize=12, fontweight='bold')
axes[0].legend(loc='best')
axes[0].grid(True, alpha=0.3)

# Plot 2: Cumulative Rewards
cumulative_rewards = np.cumsum(reward_callback.episode_rewards)
axes[1].plot(cumulative_rewards, linewidth=1.5, color='green', label='Cumulative Reward')
axes[1].set_xlabel('Episode', fontsize=11)
axes[1].set_ylabel('Cumulative Reward', fontsize=11)
axes[1].set_title('DQN Training: Cumulative Rewards Over Time', fontsize=12, fontweight='bold')
axes[1].legend(loc='best')
axes[1].grid(True, alpha=0.3)

# Save the trained model
model.save("dqn_iiser_model")

print("Model saved as 'dqn_iiser_model.zip'")

# Evaluation
print("\nEvaluating the trained model...")
obs, _ = env.reset()
done = False
total_reward = 0
steps = 0
while not done and steps < 2000:  # Limit steps to avoid infinite loop
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    done = terminated or truncated
    steps += 1

print(f"Evaluation completed in {steps} steps with total reward: {total_reward:.2f}")
env.close()

# Print Training Statistics
print(f"\nTraining Statistics:")
print(f"Total Episodes: {len(reward_callback.episode_rewards)}")
print(f"Average Reward: {np.mean(reward_callback.episode_rewards):.2f}")
print(f"Max Reward: {np.max(reward_callback.episode_rewards):.2f}")
print(f"Min Reward: {np.min(reward_callback.episode_rewards):.2f}")
print(f"Std Dev: {np.std(reward_callback.episode_rewards):.2f}")
if len(reward_callback.episode_rewards) > 100:
    print(f"Last 100 Episodes Average: {np.mean(reward_callback.episode_rewards[-100:]):.2f}")
