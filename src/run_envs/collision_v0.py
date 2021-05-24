from src.envs.collision_v0.collision import Collision
from src.envs.collision_v0.agents.dummy import Agent
import numpy as np 

def collision_v0():

    agent = Agent(400, 400, 10, 1)
    env = Collision(400, 400, 30, agent)

    for _ in range(10000):
        action = env.player.get_action(env.state)
        ob, reward, done, _ = env.step(action)
        env.render()
        if done:
            env.reset()