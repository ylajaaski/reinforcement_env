from src.envs.collision_v0.collision import Collision, Debug
from src.envs.dynamics import *
import pandas as ps
import os 

ITER = 10000 #10000
BALLS = [20, 40, 80]

def dynamics_test(csv_name):

    energy_ = []
    momentum_ = []

    e = ps.DataFrame(data={})
    m = ps.DataFrame(data={})
    for count in BALLS:
        a = Debug(800, 800, 10, (200, 10), 1)
        env = Collision(800, 800, count, a)
        for _ in range(ITER):
            action = (0,0)
            energy_.append(energy(env.balls))
            momentum_.append(momentum(env.balls))
            env.step(action)

        e[count] = energy_
        m[count] = momentum_
        energy_ = []
        momentum_ = []
    
    path = os.getcwd()
    e_path = path + "/results/tests/collision_v0/dynamics/energy/{}.csv".format(csv_name)
    m_path = path + "/results/tests/collision_v0/dynamics/momentum/{}.csv".format(csv_name)
    e.to_csv(e_path)
    m.to_csv(m_path)