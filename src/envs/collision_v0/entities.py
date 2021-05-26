import numpy as np
from src.core import Entity, Player 

PLAYER_COLOR = (31, 120, 10) # dark green

class Ball(Entity):
    
    def __init__(self, env_width, env_height, position, radius, color, timestep):
        self.env_width = env_width
        self.env_height = env_height 
        self.radius = radius
        self.color = color
        self.timestep = timestep 
        self.position = position 
        self.speed = np.array([np.random.random()*2-1, np.random.random()*2-1])
        self.changed_speed = 0

        self.left = self.position[0] - self.radius
        self.right = self.position[0] + self.radius
        self.bottom = self.position[1] + self.radius
        self.top = self.position[1] - self.radius
    
    def move(self):
        
        self.speed += self.changed_speed
        self.changed_speed = 0
        delta_x = self.speed * self.timestep
        self.position += delta_x
        if (self.position[0] < self.radius and delta_x[0] < 0) or \
            (self.position[0] > self.env_width - self.radius and delta_x[0] > 0):
            self.speed[0] *= -1
        if (self.position[1] < self.radius and delta_x[1] < 0) or \
            (self.position[1] > self.env_height - self.radius and delta_x[1] > 0):
            self.speed[1] *= -1   
    
    def update_speed(self, speed, position):
        new_pos = self.position + self.speed * self.timestep 
        new_pos_ = position + speed * self.timestep 
        if np.sum((new_pos-new_pos_)**2) < np.sum((self.position - position)**2):
            if not (position == self.position).all():
                self.changed_speed -= (self.speed - speed) @ (self.position - position) / np.sum((self.position - position)**2) * (self.position - position)

class Debug(Player):

    def __init__(self, env_width, env_height, radius, position, timestep):
        self.env_width = env_width
        self.env_height = env_height
        self.radius = radius
        self.position = position
        self.color = PLAYER_COLOR
        self.speed = np.zeros(2)
        self.max_speed = 6
        self.timestep = timestep 
    
    def valid_state(self, position, ball_list):
        return True
    
    def move(self, action, ball_list):
        position = self.position + action
        if self.valid_state(position, ball_list):
            self.position += action * self.timestep 
            return True
        else:
            return False
    
    def get_action(self, state):
        return 0
    
    def reset(self):
        return 
    
    def load_model(self):
        return
    
    def save_model(self):
        return