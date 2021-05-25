import argparse
from src.utils import listdir_nohidden

# Environment tests
from src.tests.collision_v0.collision_speed import *
from src.tests.collision_v0.test_dynamics import *
from src.tests.collision_v0.neural_networks import *

'''
Testing performance and dynamics of the environments is possible with this script. The testing scripts can
be found from <src/tests>. 

Required args:

    environment  : name of the environment
    test         : name of the test
    file         : name of the file where results are saved

Optional args:
    episodes : number of episodes
'''

ap = argparse.ArgumentParser()
ap.add_argument("-env", "--environment", required=True, help="Name of the environment e.g. collision_v0.")
ap.add_argument("-t", "--test", required = True, help="Name of the test e.g. collision_speed.")
ap.add_argument("-f", "--file", required = True, help="Name of the file where results are saved.")
ap.add_argument("-eps", "--episodes", default = 1000, help="Episodes before the training/testing ends")
args = vars(ap.parse_args())

env_path = "src/envs"
environments = [name for name in listdir_nohidden(env_path) if os.path.isdir(os.path.join(env_path, name))]
env = args["environment"]

test_path = "src/tests/{}".format(env)
tests = [name for name in os.listdir(test_path) if not (name.startswith('.') or name.startswith('_'))]
test = args["test"]
file = args["file"]

def main():
    if env not in environments:
        print("<" + env + ">" + " is not part of the possible environments:")
        print(list(environments))
    else:
        if test + ".py" not in tests:
            print("<" + test + ">" + " is not part of the possible tests:")
            print(list(tests))
        else:
            print(test)
            globals()[test](file)
        
if __name__ == "__main__":
    main()