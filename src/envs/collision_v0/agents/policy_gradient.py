from src.core import Player
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

PLAYER_COLOR = (31, 120, 10) # dark green

class Agent(Player):
    
    def __init__(self, env_width, env_height, radius, timestep):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env_width = env_width
        self.env_height = env_height
        self.radius = radius
        self.color = PLAYER_COLOR
        self.speed = np.zeros(2)
        self.max_speed = 6
        self.timestep = timestep
        self.network = Policy().to(self.device)

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

class Policy(nn.Module):

    def __init__(self):
        super(Policy, self).__init__()
        self.conv = nn.Sequential(
                        nn.Conv2d(in_channels = 1, out_channels =  20, kernel_size = 5),
                        nn.BatchNorm2d(20),
                        nn.ReLU(),
                        nn.Conv2d(in_channels = 20, out_channels =  20, kernel_size = 5),
                        nn.BatchNorm2d(20),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size = 2))

        self.fc = nn.Sequential(
                        nn.Linear(in_features = 10, out_features = 200),
                        nn.ReLU())
        
        self.x_speed = nn.Linear(in_features = 200, out_features = 6)
        self.y_speed = nn.Linear(in_features = 200, out_features = 6)

        self.value = nn.Linear(in_features = 200, out_features = 1)

    def forward(self, state):

        # Convolutional layers
        x = self.conv(state)

        # Fully connected layers
        x = x.reshape(1) #??????
        x = self.fc(x)

        # State value
        state_value = self.value(x)

        # Action
        x_speed = self.x_speed(x)
        y_speed = self.y_speed(x)

        x_probs = F.softmax(x_speed, dim = -1)
        y_probs = F.softmax(y_speed, dim = -1)
        policies = Categorical(x_probs), Categorical(y_probs)

        return policies, state_value 
