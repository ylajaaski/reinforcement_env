from src.envs.collision import Collision, Agent, Ball, Debug
import numpy as np 

def run():

    a = Debug(400, 400, 10, (100, 200), 1)
    env = Collision(400, 400, a, 30)

    for _ in range(10000):
        action = np.random.random()*12 - 6
        env.step(action)
        env.render()