import argparse
import os 
from src.utils import listdir_nohidden

# Environments
from src.run_envs.collision_v0 import *
from src.run_envs.collision_v1 import *

'''
Training and testing different environments and agents is possible with this script. 

Required args:

    environment  : name of the environment
    train (bool) : 1 used for training

Optional args:
    episodes  : number of episodes
    rendering : type of rendering can be, none, not_saved or saved
'''

ap = argparse.ArgumentParser()
ap.add_argument("-env", "--environment", required=True, help="Name of the environment e.g. collision_v0")
ap.add_argument("-t", "--train", required=True, help="You can either train or test an agent")
ap.add_argument("-eps", "--episodes", default = 1000, help="Episodes before the training/testing ends")
ap.add_argument("-r", "--rendering", default ="none", help = "You can visualize and even save the rendering as a video.")
args = vars(ap.parse_args())

env_path = "src/envs"
environments = [name for name in listdir_nohidden(env_path) if os.path.isdir(os.path.join(env_path, name))]
env = args["environment"]

def main():
    if env not in environments:
        print("<" + env + ">" + " is not part of the possible environments:")
        print(list(environments))
    else:
        print("Started...")
        globals()[env]()
        
if __name__ == "__main__":
    main()