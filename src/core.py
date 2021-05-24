class Environment(object):


    def step(self, action):

        return NotImplementedError
    
    def reset(self):

        return NotImplementedError
    
    def render(self):

        return NotImplementedError
    
    def seed(self, seeds = None):

        return

class Player(object):

    def move(self, action):

        return NotImplementedError
    
    def valid_position(self):

        return NotImplementedError

class Entity(object):

    def move(self, action = None):

        return NotImplementedError