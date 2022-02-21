import numpy as np

def flow_single(image, flow, W=None):
    """
    Flow an image using np.roll
    
    Parameters
    ----------
    image : np.ndarray of shape (x,y) or (x,y,nc)
        Image to be flowed
    flow : np.ndarray of size 2 and dtype float
        Uniform shift (di,dj)
    W : np.ndarray of shape (x,y) or (x,y,nc)
        Weights associated with image to be flowed
    
    Returns
    -------
    (result, W_result)

    result : np.ndarray of shape (x,y) or (x,y,nc)
        Image flowed
    W_result : np.ndarray of shape (x,y) or (x,y,nc)
        Weights flowed
    """

    d = image.shape[0]

    res = np.zeros_like(image)
    
    if W is None:
        W = np.ones_like(image)

    W_res = np.zeros_like(image)

    i, j = flow[0], flow[1]
    i0, j0 = int(np.floor(i)), int(np.floor(j))

    res += (i0+1-i)*(j0+1-j)*np.roll(image, shift = (i0,j0), axis = (0,1))
    res += (i-i0)*(j0+1-j)*np.roll(image, shift = (i0+1,j0), axis = (0,1))
    res += (i-i0)*(j-j0)*np.roll(image, shift = (i0+1,j0+1), axis = (0,1))
    res += (i0+1-i)*(j-j0)*np.roll(image, shift = (i0,j0+1), axis = (0,1))

    W_res += (i0+1-i)*(j0+1-j)*np.roll(W, shift = (i0,j0), axis = (0,1))
    W_res += (i-i0)*(j0+1-j)*np.roll(W, shift = (i0+1,j0), axis = (0,1))
    W_res += (i-i0)*(j-j0)*np.roll(W, shift = (i0+1,j0+1), axis = (0,1))
    W_res += (i0+1-i)*(j-j0)*np.roll(W, shift = (i0,j0+1), axis = (0,1)) 

    W_res[0:max(0,int(np.ceil(i))),:] = 0
    W_res[:,0:max(0,int(np.ceil(j)))] = 0
    W_res[min(d,d+i0):d,:] = 0
    W_res[:,min(d,d+j0):d] = 0

    return res, W_res