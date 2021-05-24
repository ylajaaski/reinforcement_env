import argparse

# Environment tests
from src.run_envs.collision_v0 import *

'''
Training and testing different environments and agents is possible with this script. 

Required args:

    environment  : name of the environment
    train (bool) : 1 used for training

Optional args:
    episodes : number of episodes
'''

ap = argparse.ArgumentParser()
ap.add_argument("-env", "--environment", required=True, help="Name of the environment e.g. collision_v0")
ap.add_argument("-t", "--train", required=True, help="You can either train or test an agent")
ap.add_argument("-eps", "--episodes", default = 1000, help="Episodes before the training/testing ends")
args = vars(ap.parse_args())

environments = {"collision_v0": collision_v0} # Possible environments
env = args["environment"]

def main():
    if env not in environments:
        print("<" + env + ">" + " is not part of the possible environments:")
        print(list(environments.keys()))
    else:
        environments[env]()
        
if __name__ == "__main__":
    main()