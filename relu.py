import numpy as np

def relu_forword(activations):
    return np.maximum(0,activations)

def relu_backword(dout,cache):
    return_value=dout
    return_value[cache<=0]=0
    
    return return_value