from src.core import Player
from src.utils import transform_frame, discount_rewards
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

PLAYER_COLOR = (31, 120, 10) # dark green

class Agent(Player):
    
    # TODO: add n_size to initialization to choose the interpolated frame size
    def __init__(self, env_width, env_height, radius, timestep):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env_width = env_width
        self.env_height = env_height
        self.radius = radius
        self.color = PLAYER_COLOR
        self.speed = np.zeros(2)
        self.max_speed = 6
        self.gamma = 0.92
        self.timestep = timestep
        self.network = Policy().to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = 0.0001)
        self.previous_frames = []
        self.rewards = []
        self.log_probs = []
        self.state_values = []
        self.actions = []
        self.entropies = []

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
        speed = action.item()
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
    
    def get_action(self, state):
        # Lowering resolution and to grayscale
        frame = self.transform_frame(state) 
        
        # Form a state from frames
        state = self.frames_to_state(frame).to(self.device)
        
        # Distribution and value from NN
        policy_distr, _ = self.network.forward(state)
        
        # Take action with the highest probability 
        action = torch.argmax(policy_distr.probs)[1] 

        return action
    
    def get_training_action(self, state):
        # Resize and grascale
        frame = transform_frame(state, 150) # TODO: n_size = 150

        state = self.frames_to_state(frame).to(self.device)

        policy_distr, state_value = self.network(state)

        # Sample an action from the policy distribution 
        action = policy_distr.sample()[1]

        log_probability = policy_distr.log_prob(action)

        # Determine entropy
        entropy = policy_distr.entropy()

        return action, log_probability, state_value, entropy

    
    def reset(self, player_position):
        self.frames = []
        self.position = player_position
    
    def save_model(self):
        #torch.save(self.network.state_dict(), "results//{:.3f}_{}.pth".format(win_rate[episode],episode))
        return
        
    def load_model(self):
        return
    
    def frames_to_state(self, current_frame):
        if not self.previous_frames:
            self.previous_frames = [current_frame]  
        frame_stack = torch.stack((self.previous_frames[0].T, current_frame.T), dim = 0)
        self.previous_frames = [current_frame]
        return frame_stack 
    
    def update_network(self):
        # Transforming the agent memory into tensors
        state_values = torch.stack(self.state_values, dim=0).squeeze().to(self.device)
        log_probs = torch.stack(self.log_probs, dim=0).squeeze().to(self.device)
        returns = discount_rewards(torch.tensor(self.rewards, device=self.device, dtype=torch.float), self.gamma)
        entropies = torch.stack(self.entropies, dim=0).squeeze().to(self.device)
        
        # Resetting the agent's memory for a new episode
        self.state_values = [] 
        self.rewards = []
        self.log_probs = []
        self.entropies = []

        # Advantage estimates:
        state_values = state_values.T[1]
        adv_ests = returns - state_values

        # Policy loss:
        log_probs = log_probs.T[1]
        l_PG = -torch.mean(adv_ests.detach() * log_probs)

        # Value loss:
        l_v = F.mse_loss(state_values, returns.detach())

        # Entropy loss:
        l_H = -torch.mean(entropies)

        # Total loss:
        loss = l_PG + l_v + l_H

        # Optimization
        loss.backward()
        nn.utils.clip_grad_norm(self.network.parameters(), .5)
        self.optimizer.step()
        self.optimizer.zero_grad()
        return l_PG.item(), l_v.item(), l_H.item()
    
    def store_step(self, state_value, reward, log_probability, entropy):
        # Adding the observed quantities to the agent's memory
        self.state_values.append(state_value)
        self.rewards.append(reward)
        self.log_probs.append(log_probability)
        self.entropies.append(entropy)


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
                        nn.Linear(in_features = 20*71*71, out_features = 200),
                        nn.ReLU())
        
        self.speed = nn.Linear(in_features = 200, out_features = 4)

        self.value = nn.Linear(in_features = 200, out_features = 1)

    def forward(self, state):

        # Convolutional layers
        x = self.conv(state)

        # Fully connected layers
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 20*71*71)
        x = self.fc(x)

        # State value
        state_value = self.value(x)

        # Action
        speed = self.speed(x)

        probs = F.softmax(speed, dim = -1)
        policies = Categorical(probs)

        return policies, state_value 
