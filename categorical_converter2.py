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

lookup_data=dict()

def lookup(key):
    global lookup_data
    if key in lookup_data:
        return lookup_data[key]
    else:        
        if not lookup_data:
            lookup_data[key]=1
            return 1
        else:
            value=max(lookup_data.values())+1
            lookup_data[key]=value
            return value

def convert(data,cols):
    global lookup_data
    N,D=data.shape
    
    for ix,i in  enumerate(cols):        
        for x in range(N):
            data[x,i]=lookup(data[x,i]) 
        lookup_data.clear()
        
    return data


if __name__=="__main__":    
    pass