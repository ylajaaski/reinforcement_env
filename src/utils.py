import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import os 

def rgb2gray(rgb):
    '''
    Transforms frame into a grayscale

    Args:
        rgb (numpy array) : numpy array grayscaled
    '''

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def transform_frame(frame, n):
    '''
    Transform the size of the frame. Note that the frame should be shaped M*M*3 (square)

    Args:
        frame (numpy array) : shaped M*M*3
        n (int)             : new size of the frame (n*n*3)
    
    Returns:
        frame (torch tensor) : shaped n*n*3

    '''
    shape = frame.shape
    assert len(shape) == 3
    assert shape[0] == shape[1] and shape[2] == 3

    color0 = frame[:,:,0]
    color1 = frame[:,:,1]
    color2 = frame[:,:,2]
    current_frame = np.array([color0, color1, color2])
    batch = torch.from_numpy(current_frame).unsqueeze(0)
    batch = F.interpolate(batch.float(), (n, n))
    small_frame = batch.squeeze(0)
    return -small_frame[0,:,:].unsqueeze(dim = -1)

def discount_rewards(r, gamma):
    '''
    Discounted sum of the rewards : r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} ...

    Args:
        r (list)      : List of rewards returned by the environment
        gamma (float) : discount factor
    
    Return:
        dr (float) : discounted sum of the rewards
    '''
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class PrintLayer(nn.Module):
    '''
    Print layer shapes inside a sequential structure
    '''
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        print(x.shape)
        return x

def listdir_nohidden(path):
    '''
    Extension to os.listdir to ignore files starting with '.' or '_'
    '''
    for f in os.listdir(path):
        if not (f.startswith('.') or f.startswith('_')):
            yield f