import argparse
from src.tests.collision_v0.collision_speed import *

# Environment tests
from src.tests.collision_v0.collision_speed import *

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

environments = {"collision_v0": {"collision_speed" : collision_test}} # Possible tests
env = args["environment"]
test = args["test"]

def main():
    if env not in environments:
        print("<" + env + ">" + " is not part of the possible environments:")
        print(list(environments.keys()))
    else:
        tests = environments[env]
        if test not in tests:
            print("<" + test + ">" + " is not part of the possible tests:")
            print(list(tests.keys()))
        else:
            tests[test](args["file"])
        
if __name__ == "__main__":
    main()