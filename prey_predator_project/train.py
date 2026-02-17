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
from stable_baselines3.common.monitor import Monitor
from utils import SingleAgentWrapper
from config import *

SAVE_EVERY_N_EPISODES = 200


class TrainingCallback(BaseCallback):
    def __init__(self, agent_name):
        super().__init__(verbose=0)
        self.agent_name = agent_name
        self.is_predator = agent_name == "predator"
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_catches = []
        self.output_dir = f"training_metrics/{agent_name}"
        self._last_save = 0
        os.makedirs(self.output_dir, exist_ok=True)

    def _on_step(self):
        for done, info in zip(self.locals.get("dones", []), self.locals.get("infos", [])):
            if done and "episode" in info:
                r, l = float(info["episode"]["r"]), int(info["episode"]["l"])
                self.episode_rewards.append(r)
                self.episode_lengths.append(l)
                if self.is_predator:
                    self.episode_catches.append(int(r > 50))

        n = len(self.episode_rewards)
        if n >= 20 and (n - self._last_save) >= SAVE_EVERY_N_EPISODES:
            self._save(n)
        return True

    def _on_training_end(self):
        if len(self.episode_rewards) >= 2:
            self._save(len(self.episode_rewards))

    def _moving_avg(self, data, window):
        return np.convolve(data, np.ones(window) / window, mode="valid")

    def _save(self, n):
        self._last_save = n
        window = min(50, max(10, n // 10))

        plots = [
            ("Rewards", self.episode_rewards, "blue", "Reward"),
            ("Episode Length", self.episode_lengths, "green", "Steps"),
        ]
        if self.is_predator and len(self.episode_catches) >= window:
            catch_rate = self._moving_avg(self.episode_catches, window) * 100
            plots.append(("Catch Rate", catch_rate, "red", "Catch Rate (%)"))

        fig, axes = plt.subplots(1, len(plots), figsize=(6 * len(plots), 5))
        if len(plots) == 1:
            axes = [axes]

        for ax, (title, data, color, ylabel) in zip(axes, plots):
            if title in ("Rewards", "Episode Length"):
                ax.plot(data, alpha=0.2, color="gray")
                ax.plot(range(window - 1, len(data)), self._moving_avg(data, window), linewidth=2, color=color)
            else:
                ax.plot(range(window - 1, len(self.episode_catches)), data, linewidth=2, color=color)
                ax.set_ylim([0, 100])
            ax.set(title=title, xlabel="Episode", ylabel=ylabel)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/progress.png", dpi=150, bbox_inches="tight")
        plt.close()

        with open(f"{self.output_dir}/stats.txt", "w") as f:
            f.write(f"Episodes: {n}\n")
            f.write(f"Timesteps: {self.num_timesteps}\n")
            f.write(f"Mean Reward: {np.mean(self.episode_rewards):.2f}\n")
            f.write(f"Mean Length: {np.mean(self.episode_lengths):.2f}\n")
            if self.is_predator and len(self.episode_catches) >= window:
                f.write(f"Catch Rate: {np.mean(self.episode_catches[-window:]) * 100:.1f}%\n")


def make_env(agent_id):
    return lambda: Monitor(SingleAgentWrapper(agent_id))


def train(agent_type, timesteps, n_envs):
    agent_id = 0 if agent_type == "predator" else 1
    print(f"\nTraining {agent_type} | Timesteps: {timesteps:,} | Envs: {n_envs}\n")

    os.makedirs("models", exist_ok=True)
    env = SubprocVecEnv([make_env(agent_id) for _ in range(n_envs)])

    model = PPO(
        "MlpPolicy", env,
        learning_rate=LEARNING_RATE,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        ent_coef=ENT_COEF,
        verbose=1,
    )
    model.learn(total_timesteps=timesteps, callback=TrainingCallback(agent_type), progress_bar=True)
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