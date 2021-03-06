class Environment(object):
    '''
    Reinforcement learning agents can be trained and tested in the underlying environment. Each
    environment has its own individual characteristics. 

    Every environment is used to train a Player (see below). If no Player is defined, the constructed
    environment can be used for debugging purposes, such as performance and environment dynamics analysis.

    Methods:

        step
        reset
        render
        seed
    '''


    def step(self, action):
        '''
        Run one timestep of the dynamics of the environment. Changes the state of the environment and
        the state of the defined Player.

        Args:
            action (object) : An action provided by the Player.
        
        Returns:
            observation (object) : The next state of the environment (observable by the Player)
            reward (float)       : Reward caused by the previous action
            done (bool)          : Determines wheter an episode has ended (e.g. invalid position of the Player)
            info (dict)          : Performance and environment dynamics analysis

        '''
        raise NotImplementedError
    
    def reset(self):
        '''
        Initializes a new instance of the environment which is independent of the previous states. The new
        instance is always a valid position (done = False). Using seed (see below) resets the environment
        always to the same state.
        '''
        raise NotImplementedError
    
    def render(self):
        '''
        Visualize the environment. This is usally achieved with opencv-python.
        '''
        raise NotImplementedError
    
    def seed(self, seeds = {}):
        '''
        Set seed to different random number generators for debugging purposes.

        Args:
            seeds (dict) : "numpy", "torch", "gpflow" 
        '''
        if "numpy" not in seeds:
            seeds["numpy"] = None
        if "torch" not in seeds:
            seeds["torch"] = None
        if "gpflow" not in seeds:
            seeds["gpflow"] = None

        return seeds
    
class Player(object):
    '''
    Each environment has a player that takes some actions to gather rewards.

    Note: Using environment for debugging the performance and underlying dynamics, see method
    valid_position
    '''

    def move(self, action):
        '''
        Given an action change the state of the Player. This does not change the state of the environment.
        Player must always be moved within the step function of the environment.

        Args:
            action (object) : An action taken by the Player
        '''

        raise NotImplementedError
    
    def valid_state(self):
        '''
        Determines wheter the state of the Player is valid. If you want to debug the environment, you
        might want to always return True.
        '''
        raise NotImplementedError
    
    def get_action(self, state):
        '''
        Given a state determines the next action of the Player which is used during testing.

        Args:
            state (object) : state of the agent e.g. sequence of frames
        
        Returns:
            action (object) : an action that is used to move the agent
        '''
        raise NotImplementedError
    
    def get_training_action(self, state):
        '''
        Given a state determines the next action of the Player which is used during training.

        Args:
            state (object) : state of the agent e.g. sequence of frames
        
        Returns:
            action (object)     : an action that is used to move the agent
            state_value (float) : value for the state
        '''
        raise NotImplementedError

    
    def reset(self):
        '''
        Resets the state of the player
        '''
        raise NotImplementedError
    
    def save_model(self):
        '''
        Saves the information about the agent network.
        '''
        raise NotImplementedError
    
    def load_model(self):
        '''
        Loads the information about the saved agent network.
        '''
        raise NotImplementedError
        

class Entity(object):
    '''
    Environment might include some other entities that move around. These entities might also take some
    actions. Note that entities are never trained. 
    '''

    def move(self, action = None):
        '''
        Given an action change the state of the Entity. This does not change the state of the environment.
        Entity dynamics must be defined within the environment, state validity is not checked.

        Args:
            action (object) : An action taken by the Player
        '''
        return NotImplementedError