import numpy as np
import torch
import torch.nn.functional as F

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
        frame (numpy array) : shaped n*n*3

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
    small_frame = np.array(small_frame)
    return np.expand_dims(-small_frame[0,:,:], axis = -1)