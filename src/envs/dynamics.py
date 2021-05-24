import numpy as np

def energy(entities):
    '''
    Calculates the kinetic energy of the system excluding the agent.

    Args:
        entities (list) : list of entities in the environment
    
    Returns:
        energy (float) : kinetic energy
    '''
    speeds = np.array(list(map(lambda x: x.speed, entities))).flatten()
    return np.sum(speeds**2)

def momentum(entities):
    '''
    Calculates the overall momentum of the environment exclduing the agent.

    Args:
        entities (list) : list of entities in the environment

    Returns:
        momentum (float) : momentum of the environment

    '''
    speeds = np.array(list(map(lambda x: x.speed, entities))).flatten()
    return np.sum(speeds)