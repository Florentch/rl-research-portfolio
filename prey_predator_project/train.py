"""Training script"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from utils import SingleAgentWrapper
from config import *


class TrainingCallback(BaseCallback):
    def __init__(self, agent_name):
        super().__init__(verbose=0)
        self.agent_name = agent_name
        self.is_predator = (agent_name == "predator")
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_catches = []
        self.output_dir = f"training_metrics/{agent_name}"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _on_step(self):
        for done, info in zip(self.locals.get('dones', []), self.locals.get('infos', [])):
            if done and 'episode' in info:
                self.episode_rewards.append(float(info['episode']['r']))
                self.episode_lengths.append(int(info['episode']['l']))
                if self.is_predator:
                    self.episode_catches.append(1 if info['episode']['r'] > 50 else 0)
        
        if self.num_timesteps % SAVE_FREQ == 0 and len(self.episode_rewards) >= 20:
            self._save_plots()
        return True
    
    def _save_plots(self):
        window = min(50, max(10, len(self.episode_rewards) // 10))
        ma_rewards = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
        ma_lengths = np.convolve(self.episode_lengths, np.ones(window)/window, mode='valid')
        
        if self.is_predator:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].plot(self.episode_rewards, alpha=0.2, color='gray')
        axes[0].plot(range(window-1, len(self.episode_rewards)), ma_rewards, linewidth=2, color='blue')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Rewards')
        axes[0].grid(alpha=0.3)
        
        axes[1].plot(self.episode_lengths, alpha=0.2, color='gray')
        axes[1].plot(range(window-1, len(self.episode_lengths)), ma_lengths, linewidth=2, color='green')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Steps')
        axes[1].set_title('Episode Length')
        axes[1].grid(alpha=0.3)
        
        if self.is_predator and len(self.episode_catches) >= window:
            catch_rate = np.convolve(self.episode_catches, np.ones(window)/window, mode='valid') * 100
            axes[2].plot(range(window-1, len(self.episode_catches)), catch_rate, linewidth=2, color='red')
            axes[2].set_xlabel('Episode')
            axes[2].set_ylabel('Catch Rate (%)')
            axes[2].set_title('Catch Rate')
            axes[2].set_ylim([0, 100])
            axes[2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/progress.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        with open(f'{self.output_dir}/stats.txt', 'w') as f:
            f.write(f"Episodes: {len(self.episode_rewards)}\n")
            f.write(f"Timesteps: {self.num_timesteps}\n")
            f.write(f"Mean Reward: {np.mean(self.episode_rewards):.2f}\n")
            f.write(f"Mean Length: {np.mean(self.episode_lengths):.2f}\n")
            if self.is_predator and len(self.episode_catches) >= window:
                f.write(f"Catch Rate: {np.mean(self.episode_catches[-window:])*100:.1f}%\n")
    
    def _on_training_end(self):
        if len(self.episode_rewards) >= 20:
            self._save_plots()


def make_env(agent_id):
    return lambda: SingleAgentWrapper(agent_id)


def train(agent_type, timesteps, n_envs):
    agent_id = 0 if agent_type == "predator" else 1
    
    print(f"\nTraining {agent_type}")
    print(f"Timesteps: {timesteps:,} | Envs: {n_envs}\n")
    
    os.makedirs("models", exist_ok=True)
    env = SubprocVecEnv([make_env(agent_id) for _ in range(n_envs)])
    callback = TrainingCallback(agent_name=agent_type)
    
    model = PPO(
        "MlpPolicy", env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        ent_coef=ENT_COEF,
        verbose=1
    )
    
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)
    model.save(f"models/{agent_type}_final")
    
    print(f"\nSaved: models/{agent_type}_final.zip\n")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train agent")
    parser.add_argument("agent", choices=["predator", "prey"])
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--n-envs", type=int, default=DEFAULT_N_ENVS)
    args = parser.parse_args()
    train(args.agent, args.timesteps, args.n_envs)
