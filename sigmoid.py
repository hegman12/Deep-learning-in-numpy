import numpy as np

def sigmoid_forword(activations):
    #print("sigmoid FW",activations.shape)
    return 1/1+np.exp(activations)

def sigmoid_backword(dout,sigmoid_activation):
    #print("sigmoid BK",dout.shape,sigmoid_activation.shape)
    return dout * ((sigmoid_activation) * (1-sigmoid_activation))