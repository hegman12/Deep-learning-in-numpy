import numpy as np
import batch_norm as bn

def eval_numerical_gradient(f, x, cache):
  """ 
  a naive implementation of numerical gradient of f at x 
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """ 

  grad = np.zeros(x.shape)
  h = 0.00001

  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index
    old_value = x[ix]
    x[ix] = old_value + h # increment by h
    fxhp[ix] = f(x,cache) # evalute f(x + h)
    x[ix] = old_value # restore to previous value (very important!)

    # compute the partial derivative

    it.iternext() # step to next dimension
    
      # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index
    old_value = x[ix]
    x[ix] = old_value - h # increment by h
    fxhm[ix] = f(x,cache) # evalute f(x + h)
    x[ix] = old_value # restore to previous value (very important!)

  grad = (fxh - fxhm) / (2*h) # the slope


  return grad