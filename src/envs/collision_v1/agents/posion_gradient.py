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
        self.gamma = 0.99
        self.timestep = timestep
        self.network = Policy(62,4).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr = 0.0002)
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
        #frame = self.transform_frame(state) 
        
        # Form a state from frames
        #state = self.frames_to_state(frame).to(self.device)
        
        # Distribution and value from NN
        policy_distr, _ = self.network.forward(state)
        
        # Take action with the highest probability 
        action = torch.argmax(policy_distr.probs)

        return action
    
    def get_training_action(self, state):
        # Resize and grascale
        #frame = transform_frame(state, 150) # TODO: n_size = 150
        state = torch.from_numpy(state).float()
        state = self.frames_to_state(state).to(self.device)
        #print(state.shape)

        policy_distr, state_value = self.network(state)

        # Sample an action from the policy distribution 
        action = policy_distr.sample()

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
            self.previous_frames = [current_frame, current_frame, current_frame]  
        frame_stack = torch.stack((self.previous_frames[0].T, self.previous_frames[1].T, self.previous_frames[2].T, current_frame.T), dim = 0)
        self.previous_frames = [self.previous_frames[1], self.previous_frames[2], current_frame]
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
        #state_values = state_values 
        adv_ests = returns - state_values

        # Policy loss:
        log_probs = log_probs 
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
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.fc1 = torch.nn.Linear(state_space, 200)
        self.fc2 = torch.nn.Linear(200,200)
        self.fc3 = torch.nn.Linear(4*200,200)
        self.fc4 = torch.nn.Linear(200, action_space)

        self.value = torch.nn.Linear(200,1)
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
        #shape_ = x.shape
        x = x.reshape(1, 200*4)
        x = self.fc3(x) 
        x = F.relu(x)
        y = self.fc4(x)
        probs = F.softmax(y, dim=-1)
        policies = Categorical(probs)

        values = self.value(x)
        return policies, values 

    
    
    
    
    
    
    
    
    
# def forward(self, state):

#     # Convolutional layers
#     x = self.conv(state)

#     # Fully connected layers
#     #batch_size = x.shape[0]
#     x = x.reshape(1, 4*10*11*11)
#     x = self.fc(x)

#     # State value
#     state_value = self.value(x)

#     # Action
#     speed = self.speed(x)

#     probs = F.softmax(speed, dim = -1)
#     policies = Categorical(probs)

#     return policies, state_value 