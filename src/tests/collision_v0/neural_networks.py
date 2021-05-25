from src.envs.collision_v0.collision import Collision_v0
from src.envs.collision_v0.agents.policy_gradient import Agent
from src.utils import transform_frame, rgb2gray
import torch

def neural_networks(file):
    agent = Agent(400,400,10,1)
    env = Collision_v0(400,400,20,agent)

    state = env.state 
    state = transform_frame(state, 150)
    state = rgb2gray(state).unsqueeze(dim = 0)
    policy, state_value = agent.network(state.unsqueeze(dim = 0))
    print("State:", state_value)
    x_speed = torch.argmax(policy[0].probs)
    y_speed = torch.argmax(policy[1].probs)
    print("x:", x_speed, "y:", y_speed)
