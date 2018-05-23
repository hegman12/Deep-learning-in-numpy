import numpy as np

def batch_norm_forword(activations,gamma,beta):
    
    N,D=activations.shape # 2x2
    
    sum=np.sum(activations,axis=0) # 1x2
    mean = sum/N # 1x2
    var = activations - mean # 2x2 - 1x2 =  2x2
    varsqr = np.sum(np.square(var),axis=0)/N # 1x2
        
    stddev = np.sqrt(varsqr) #1x2
    
    inv_sd = 1/stddev #1x2
    
    nacct = inv_sd * var # 1x2 * 2x2 = 2x2
    
    temp=nacct*gamma
    out=temp+beta
    
    return out,(var,inv_sd,stddev,varsqr,nacct,gamma)

def batch_norm_backword(dout,cache):
    
    var,inv_sd,stddev,varsqr,nacct,gamma = cache
    N,D=dout.shape #ch
    #print(dout.T.shape,"batch norm")
    
    dtemp=dout * 1 # 2x2
    dbeta=np.sum(dout,axis=0) * 1 # ssum because the shape of beta is (1,D) # 1x2  #ch
    
    dgamma = np.sum(nacct * dtemp,axis=0) # 1x2
    dnacct = gamma * dtemp # 2x2
    
    dinv_sd = np.sum(var * dnacct,axis=0) # 1x2
    dvariance_1=inv_sd * dnacct # 2x2
    
    dstddev = -1 * (1/np.square(stddev)) * dinv_sd # 1x2
    dvarsqr = dstddev * 0.5 * (1/np.sqrt(varsqr)) # 1x2
    
    dvarience_2= 2 * var * (1/N) * dvarsqr # 2x2
    
    dvarience= dvariance_1 + dvarience_2 #2x2
    dacct_1=dvarience * 1
    
    dmean = 1/N * np.sum(-1 * dvarience,axis=0)
    
    dacct_2=np.ones((N,D))* dmean.reshape(1,D)
    
    dx=dacct_1+dacct_2 # 1x2 + 2x2 = 2x2
    
    
    return dx,dgamma,dbeta
    
    
    
    
    
    
    
    
    
    
    
    
    
