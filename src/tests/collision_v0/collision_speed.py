from src.envs.collision_v0.collision import Collision, Ball, Debug
import time 
import pandas as ps
import os 

ITER = 1000 #10000
BALLS = [10, 20, 40, 80, 160]

def collision_test(csv_name):

    times = ps.DataFrame(data={})
    for count in BALLS:
        a = Debug(800, 800, 10, (200, 10), 1)
        env = Collision(800, 800, count, a)
        start = time.time()
        for _ in range(ITER):
            action = (0,0)
            env.step(action)
        end = time.time()
        times[count] = [end - start]
    
    path = os.getcwd()
    save_path = path + "/results/tests/collision_v0/speed/{}.csv".format(csv_name)
    times.to_csv(save_path)



    

