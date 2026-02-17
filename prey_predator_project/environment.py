"""Predator-Prey Environment"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
import os
from config import *


BG_COLOR     = (30, 30, 40)
GRID_COLOR   = (45, 45, 58)
PRED_COLOR   = (220, 80, 80)
PRED_BORDER  = (160, 40, 40)
PREY_COLOR   = (80, 200, 140)
PREY_BORDER  = (40, 150, 90)
CAUGHT_COLOR = (100, 100, 110)
TEXT_COLOR   = (180, 180, 200)
STATUS_H     = 28


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
    
    def _draw_agent(self, cx, cy, r, color, border):
        pygame.draw.circle(self.window, border, (cx, cy), r)
        pygame.draw.circle(self.window, color, (cx, cy), r - 3)
    
    def _draw_predator(self, cx, cy, r):
        outer = [(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)]
        inner = [(cx, cy - r + 4), (cx + r - 4, cy), (cx, cy + r - 4), (cx - r + 4, cy)]
        pygame.draw.polygon(self.window, PRED_BORDER, outer)
        pygame.draw.polygon(self.window, PRED_COLOR, inner)
    
    def render(self):
        if self.render_mode is None:
            return None
        
        if self.window is None:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            pygame.init()
            self.window = pygame.Surface((GRID_SIZE * CELL_SIZE, GRID_SIZE * CELL_SIZE + STATUS_H))
        
        self.window.fill(BG_COLOR)
        
        # Status bar
        font = pygame.font.Font(None, 22)
        label = f"Step {self.step_count}/{MAX_STEPS}   {'CAUGHT' if self.prey_caught else 'Alive'}"
        self.window.blit(font.render(label, True, TEXT_COLOR), (10, 6))
        
        # Grid
        for i in range(GRID_SIZE + 1):
            pygame.draw.line(self.window, GRID_COLOR, (i*CELL_SIZE, STATUS_H), (i*CELL_SIZE, GRID_SIZE*CELL_SIZE + STATUS_H))
            pygame.draw.line(self.window, GRID_COLOR, (0, i*CELL_SIZE + STATUS_H), (GRID_SIZE*CELL_SIZE, i*CELL_SIZE + STATUS_H))
        
        r = int(CELL_SIZE * 0.38)
        
        def center(pos):
            return int((pos[0] + 0.5) * CELL_SIZE), int((pos[1] + 0.5) * CELL_SIZE) + STATUS_H
        
        # Prey first, predator on top
        px, py = center(self.positions[1])
        if self.prey_caught:
            self._draw_agent(px, py, r, CAUGHT_COLOR, (70, 70, 80))
        else:
            self._draw_agent(px, py, r, PREY_COLOR, PREY_BORDER)
        
        dx, dy = center(self.positions[0])
        self._draw_predator(dx, dy, r)
        
        return np.transpose(np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2))
    
    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None