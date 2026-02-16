"""Utilities"""

import numpy as np
import gymnasium as gym
from environment import PredatorPreyEnv
from config import *


class SingleAgentWrapper(gym.Env):
    def __init__(self, agent_id):
        super().__init__()
        self.agent_id = agent_id
        self.env = PredatorPreyEnv()
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.current_observations = None
        self.last_distance = None
    
    def reset(self, **kwargs):
        observations, info = self.env.reset(**kwargs)
        self.current_observations = observations
        self.last_distance = None
        return observations[self.agent_id], info
    
    def step(self, action):
        actions = [action if i == self.agent_id else self.env.action_space.sample() for i in range(2)]
        observations, rewards, terminated, truncated, info = self.env.step(actions)
        self.current_observations = observations
        reward = rewards[self.agent_id]
        
        pred_pos = self.env.positions[0]
        prey_pos = self.env.positions[1]
        distance = np.linalg.norm(pred_pos - prey_pos)
        max_dist = np.sqrt(2) * GRID_SIZE
        
        if self.agent_id == 0:
            proximity = (1 - distance / max_dist) * 50
            if self.last_distance is not None:
                proximity += (self.last_distance - distance) * 20
            if distance < 2:
                proximity += 30
            reward += proximity
        else:
            escape = (distance / max_dist) * 10
            if distance < 3:
                escape -= 5
            reward += escape
        
        self.last_distance = distance
        return observations[self.agent_id], reward, terminated, truncated, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        self.env.close()


def load_model(path):
    if path is None or path == "random":
        return None
    from stable_baselines3 import PPO
    try:
        return PPO.load(path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def get_action(model, observation):
    if model is None:
        return np.random.randint(0, 5)
    action, _ = model.predict(observation, deterministic=True)
    return int(action)
