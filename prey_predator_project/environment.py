"""Predator-Prey Environment"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import os
from config import *


class PredatorPreyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": RENDER_FPS}
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        self.render_mode = render_mode
        self.obstacles = set(OBSTACLES)
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=GRID_SIZE, shape=(8,), dtype=np.float32)
        self.positions = None
        self.step_count = 0
        self.prey_caught = False
        self.window = None
        self.predator_sprite = None
        self.prey_sprite = None
        
    def _is_valid_position(self, pos):
        x, y = pos
        return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and tuple(pos) not in self.obstacles
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.positions = []
        for _ in range(2):
            while True:
                pos = self.np_random.integers(0, GRID_SIZE, size=2)
                if self._is_valid_position(pos) and not any(np.array_equal(pos, p) for p in self.positions):
                    self.positions.append(pos)
                    break
        self.positions = np.array(self.positions)
        self.step_count = 0
        self.prey_caught = False
        return self._get_observations(), {"step": 0, "caught": False}
    
    def _get_observations(self):
        pred_pos, prey_pos = self.positions[0], self.positions[1]
        distance = np.linalg.norm(pred_pos - prey_pos)
        delta = prey_pos - pred_pos
        
        predator_obs = np.array([
            pred_pos[0]/GRID_SIZE, pred_pos[1]/GRID_SIZE,
            prey_pos[0]/GRID_SIZE, prey_pos[1]/GRID_SIZE,
            distance/GRID_SIZE, delta[0]/GRID_SIZE, delta[1]/GRID_SIZE, 0.0
        ], dtype=np.float32)
        
        prey_obs = np.array([
            prey_pos[0]/GRID_SIZE, prey_pos[1]/GRID_SIZE,
            pred_pos[0]/GRID_SIZE, pred_pos[1]/GRID_SIZE,
            distance/GRID_SIZE, -delta[0]/GRID_SIZE, -delta[1]/GRID_SIZE, 0.0
        ], dtype=np.float32)
        
        return [predator_obs, prey_obs]
    
    def step(self, actions):
        self.step_count += 1
        action_map = {0: [0,0], 1: [0,-1], 2: [0,1], 3: [-1,0], 4: [1,0]}
        
        for i, action in enumerate(actions):
            if i == 0 or not self.prey_caught:
                new_pos = self.positions[i] + action_map[action]
                if self._is_valid_position(new_pos):
                    self.positions[i] = new_pos
        
        rewards = [-1.0, 1.0]
        if np.array_equal(self.positions[0], self.positions[1]) and not self.prey_caught:
            self.prey_caught = True
            rewards = [99.0, -100.0]
        
        terminated = self.prey_caught
        truncated = self.step_count >= MAX_STEPS
        if truncated and not self.prey_caught:
            rewards[1] += 50
        
        return self._get_observations(), rewards, terminated, truncated, {"step": self.step_count, "caught": self.prey_caught}
    
    def _create_sprite(self, color, shape):
        sprite = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
        if shape == "predator":
            points = [(CELL_SIZE*0.2, CELL_SIZE*0.2), (CELL_SIZE*0.8, CELL_SIZE*0.5), (CELL_SIZE*0.2, CELL_SIZE*0.8)]
            pygame.draw.polygon(sprite, color, points)
        else:
            pygame.draw.circle(sprite, color, (CELL_SIZE//2, CELL_SIZE//2), int(CELL_SIZE*0.35))
        return sprite
    
    def render(self):
        if self.render_mode is None:
            return None
        
        if self.window is None:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            pygame.init()
            self.window = pygame.Surface((GRID_SIZE*CELL_SIZE, GRID_SIZE*CELL_SIZE))
            self.predator_sprite = self._create_sprite(COLOR_PREDATOR, "predator")
            self.prey_sprite = self._create_sprite(COLOR_PREY, "prey")
        
        self.window.fill((255, 255, 255))
        
        for i in range(GRID_SIZE + 1):
            pygame.draw.line(self.window, COLOR_GRID, (i*CELL_SIZE, 0), (i*CELL_SIZE, GRID_SIZE*CELL_SIZE))
            pygame.draw.line(self.window, COLOR_GRID, (0, i*CELL_SIZE), (GRID_SIZE*CELL_SIZE, i*CELL_SIZE))
        
        prey_pos = self.positions[1]
        prey_rect = self.prey_sprite.get_rect(center=(int((prey_pos[0]+0.5)*CELL_SIZE), int((prey_pos[1]+0.5)*CELL_SIZE)))
        sprite = self._create_sprite(COLOR_CAUGHT if self.prey_caught else COLOR_PREY, "prey")
        self.window.blit(sprite, prey_rect)
        
        pred_pos = self.positions[0]
        pred_rect = self.predator_sprite.get_rect(center=(int((pred_pos[0]+0.5)*CELL_SIZE), int((pred_pos[1]+0.5)*CELL_SIZE)))
        self.window.blit(self.predator_sprite, pred_rect)
        
        font = pygame.font.Font(None, 24)
        text = font.render(f"Step: {self.step_count}/{MAX_STEPS} | Caught: {self.prey_caught}", True, (0,0,0))
        self.window.blit(text, (10, 10))
        
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2))
    
    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None
