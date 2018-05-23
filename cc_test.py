import categorical_converter2 as cc
import numpy as np

data=np.genfromtxt("C:\\MLDatabases\\iris\\bank-Full.csv",delimiter=";",dtype="str")
data=np.delete(data,0,0)
data=np.delete(data,11,1)


"""
data=np.array(np.arange(100).reshape(10,10)%3,dtype=np.str) 
np.place(data,np.array(data,dtype=np.int)>-1,['a','b','c']) 

print(data)
"""
print(cc.convert(data,(1,2,3,4,5,6,7,8,9,13,14,15)))

