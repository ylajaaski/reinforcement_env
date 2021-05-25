from src.envs.collision_v0.collision import Collision_v0
from src.envs.collision_v0.agents.policy_gradient import Agent

def neural_networks(file):
    agent = Agent(400,400,10,1)
    env = Collision_v0(400,400,20,agent)

    state = env.state 
    print("Shape:", state.shape)
    agent.network(state)
