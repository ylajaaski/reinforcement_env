from src.spaces.space import Space 
import numpy as np

class Discrete(Space):

    def __init__(self, n):
        self.n = n 
    
    def sample(self):
        return np.random.randint(self.n)
    
