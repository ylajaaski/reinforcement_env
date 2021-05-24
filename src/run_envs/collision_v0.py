from src.envs.collision_v0.collision import Collision, Agent, Ball, Debug
import numpy as np 

def collision_v0():

    a = Agent(400, 400, 10, (100, 200), 1)
    env = Collision(400, 400, 30, a)

    for _ in range(10000):
        action = np.random.random(2)*12 - 6
        ob, reward, done, _ = env.step(action)
        env.render()
        if done:
            env.reset()