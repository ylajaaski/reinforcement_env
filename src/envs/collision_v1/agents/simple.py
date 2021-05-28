from src.envs.collision_v0.entities import PLAYER_COLOR
import torch
import torch.nn.functional as F
from src.utils import discount_rewards
import torch.nn as nn
import numpy as np

PLAYER_COLOR = (31, 120, 10) # dark green

class Policy(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.fc1 = torch.nn.Linear(state_space, 200)
        self.fc2 = torch.nn.Linear(200,200)
        self.fc3 = torch.nn.Linear(200, action_space)
        #self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight, 0, 1e-1)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x) 
        return F.softmax(x, dim=-1)


class Agent(object):
    # TODO: add n_size to initialization to choose the interpolated frame size
    def __init__(self, env_width, env_height, radius, timestep):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env_width = env_width
        self.env_height = env_height
        self.radius = radius
        self.color = PLAYER_COLOR
        self.speed = np.zeros(2)
        self.max_speed = 6
        self.gamma = 0.98
        self.timestep = timestep
        self.network = Policy(62,4).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = 1e-3)
        self.previous_frames = []
        self.rewards = []
        self.log_probs = []
        self.state_values = []
        self.actions = []
        self.entropies = []
        self.observations = []
        self.batch_size = 1

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
        speed = action
        if speed == 0:
            action = np.array([0,2])
        elif speed == 1:
            action = np.array([2,0])
        elif speed == 2:
            action = np.array([-2,0])
        else:
            action = np.array([0,-2])

        position = self.position + action * self.timestep 
        if self.valid_state(position, ball_list):
            self.position = position
            return True
        else:
            return False

    def episode_finished(self, episode_number):
        all_actions = torch.stack(self.actions, dim=0).to(self.device).squeeze(-1)
        all_rewards = torch.stack(self.rewards, dim=0).to(self.device).squeeze(-1)
        self.observations, self.actions, self.rewards = [], [], []
        discounted_rewards = discount_rewards(all_rewards, self.gamma)
        discounted_rewards -= torch.mean(discounted_rewards)
        discounted_rewards /= torch.std(discounted_rewards)

        weighted_probs = all_actions * discounted_rewards
        loss = torch.mean(weighted_probs)
        loss.backward()

        if (episode_number+1) % self.batch_size == 0:
            self.update_policy()

    def update_policy(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def get_action(self, observation, evaluation=False):
        x = torch.from_numpy(observation).float().to(self.device) / 400
        #min_v = torch.min(x)
        #range_v = torch.max(x) - min_v
        #x = (x - min_v) / range_v
        #print(max(x), min(x))
        #print(x)
        aprob = self.network.forward(x)
        #print(aprob)
        if evaluation:
            action = torch.argmax(aprob).item()
        else:
            dist = torch.distributions.Categorical(aprob)
            action = dist.sample().item()
        return action, aprob

    def store_outcome(self, observation, action_output, action_taken, reward):
        dist = torch.distributions.Categorical(action_output)
        action_taken = torch.Tensor([action_taken]).to(self.device)
        log_action_prob = -dist.log_prob(action_taken)

        self.observations.append(observation)
        self.actions.append(log_action_prob)
        self.rewards.append(torch.Tensor([reward]))
    
    def reset(self, player_position):
        self.position = player_position