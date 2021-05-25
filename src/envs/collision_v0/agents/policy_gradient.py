from src.core import Player
import numpy as np 

PLAYER_COLOR = (31, 120, 10) # dark green

class Agent(Player):
    
    def __init__(self, env_width, env_height, radius, timestep):
        self.env_width = env_width
        self.env_height = env_height
        self.radius = radius
        self.color = PLAYER_COLOR
        self.speed = np.zeros(2)
        self.max_speed = 6
        self.timestep = timestep

        player_position = np.array([np.random.random()*(env_width - 2*radius)+radius, np.random.random()*(env_height - 2*radius)+radius])
        self.position = player_position
    
    def valid_state(self, position, ball_list):
        positions = np.array(list(map(lambda x: x.position, ball_list)))
        sizes = np.array(list(map(lambda x: x.radius, ball_list)))
        stacked_positions = np.stack([self.position for _ in range(positions.shape[0])], axis = 1)
        if np.any(np.sum((stacked_positions - positions.T)**2, axis = 0) - (self.radius + sizes)**2 < 0):
            return False
        elif position[0] < self.radius or position[0] > self.env_width - self.radius or \
             position[1] < self.radius or position[1] > self.env_height - self.radius:
            return False
        else:
            return True
    
    def move(self, action, ball_list):
        position = self.position + action
        if self.valid_state(position, ball_list):
            self.position += action * self.timestep 
            return True
        else:
            return False
    
    def get_action(self, state):
        return np.random.random(2)*12 - 6
    
    def reset(self):
        return