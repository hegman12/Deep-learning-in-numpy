import numpy as np
import gc as gc

"""
USAGE: conver(data,cols)
        data - numpy array of data
        cols - tuple of columns to process. These columns should be categorical columns. 
        IMP: Indexing of colum in data starts with 0. Ypou cant index last column.
        
        Ex: you want to index second col here, then
        
        data
        a b c
        a b c
        x y z
        
        cols=(1,)
        
        if you want to index 1st and second, then
        
        cols=(0,1)
        
        All 3
        
        cols=(0,1,2)
        
        You can also skip numeric column, which you dont want to encode, like
        
        cols=(0,2) will skip 1 col

"""

def lookupBuilder(strArray):
    a=np.arange(len(strArray))+1
    lookups={k:v for (k,v) in zip(strArray,a)}
    return lookups

def convert(data,cols):    
            
    for ix,i in  enumerate(cols):
        col=data[:,i:i+1]
        lookup_data=lookupBuilder(np.unique(col))
        
        for idx,value in enumerate(col):
            col[idx]=lookup_data[value[0]]
        
        np.delete(data,i,1)
        gc.collect()
        np.insert(data,i,col,axis=1)  
        
    return data


if __name__=="__main__":    
    pass