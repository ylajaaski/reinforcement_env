from src.envs.collision_v0.collision import Collision_v0
from src.envs.collision_v0.agents.policy_gradient import Agent
from src.utils import transform_frame, rgb2gray
import torch
import cv2 as cv
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt 


def neural_networks(file):
    agent = Agent(400,400,10,1)
    env = Collision_v0(400,400,20,agent)

    state = env.state 
    state = np.array(state)
    state = transform_frame(state, 150)
    plt.imshow(state)
    plt.show()
    state = state.permute(2,0,1)
    state = torch.stack((state, state), dim = 0)
    policy, state_value = agent.network(state)
    print("State:", state_value)
    x_speed = torch.argmax(policy[0].probs)
    y_speed = torch.argmax(policy[1].probs)
    print("x:", x_speed, "y:", y_speed)
