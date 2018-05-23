import numpy as np
import os
import glob
from PIL import Image
from scipy import misc

def instr(str,substr,pos):
    t=[]
    counter=0
    for s in str:
        if s==substr:
            t.append(counter)
        counter += 1
    return t[pos-1]
    

def power_plant_data_regression(do_normalize):
    
    FILE="C:\\MLDatabases\\data\\uci\\power_plant\\CCPP\\Folds5x2_pp.csv"
    
    data=np.loadtxt(FILE,dtype=np.float,delimiter=",",skiprows=1)
    
    x=data[:,0:4]
    y=data[:,4]
    
    if do_normalize:
        x=x-np.mean(x,axis=1,keepdims=True)
        x=x/np.std(x,axis=1,keepdims=True)
    
    x_train=x[0:8000,:]
    y_train=y[0:8000]
    x_test=x[8000:None,:]
    y_test=y[8000:None]
    
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    
    return (x_train,y_train,x_test,y_test)


def epileptic_EEG_classification(do_normalize):
    
    """
    
    https://archive.ics.uci.edu/ml/datasets/Epileptic+Seizure+Recognition
    
    """
    
    FILE="C:\\MLDatabases\\data\\uci\\epileptic\\data.csv"
    
    data=np.loadtxt(FILE,dtype=">U",delimiter=",",skiprows=1)    
    data=np.asarray(data[:,1:None],dtype=np.float)
    
    x=data[:,0:178]
    y=data[:,178]
    
    y[y>1]=0
    
    if do_normalize:
        x=x-np.mean(x,axis=1,keepdims=True)
        x=x/np.std(x,axis=1,keepdims=True)
    
    x_train=x[0:10000,:]
    y_train=y[0:10000]
    x_test=x[10000:None,:]
    y_test=y[10000:None]
    
    print(x_train.shape,np.unique(y_train),x_test.shape,y_test.shape)
    
    return (x_train,y_train,x_test,y_test)


def energy_efficiency_regression_y1(do_normalize):
    
    """
    
    https://archive.ics.uci.edu/ml/datasets/Energy+efficiency
    
    """
    
    
    FILE="C:\\MLDatabases\\data\\uci\\energy efficiency\\ENB2012_data.csv"
    
    data=np.loadtxt(FILE,dtype=np.float,delimiter=",",skiprows=1)    
    
    x=data[:,0:8]
    y=data[:,8]
    
    if do_normalize:
        x=x-np.mean(x,axis=1,keepdims=True)
        x=x/np.std(x,axis=1,keepdims=True)
    
    x_train=x[0:668,:]
    y_train=y[0:668]
    x_test=x[668:None,:]
    y_test=y[668:None]
    
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    
    return (x_train,y_train,x_test,y_test)    

def energy_efficiency_regression_y2(do_normalize):
    
    """
    
    https://archive.ics.uci.edu/ml/datasets/Energy+efficiency
    
    """
    
    FILE="C:\\MLDatabases\\data\\uci\\energy efficiency\\ENB2012_data.csv"
    
    data=np.loadtxt(FILE,dtype=np.float,delimiter=",",skiprows=1)
    
    x=data[:,0:8]
    y=data[:,9]
    
    if do_normalize:
        x=x-np.mean(x,axis=1,keepdims=True)
        x=x/np.std(x,axis=1,keepdims=True)
    
    x_train=x[0:668,:]
    y_train=y[0:668]
    x_test=x[668:None,:]
    y_test=y[668:None]
    
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    
    return (x_train,y_train,x_test,y_test)


def spam_notspam_youtube_rnn_classification(x_onehot_encode):
    
    """
    https://archive.ics.uci.edu/ml/datasets/YouTube+Spam+Collection
    
    """
    
    x=[]
    y=[]
    unique_chars=set()
    max_len=0
    char_to_idx=dict()
    idx_to_chr=dict()
    
    FILE="C:\\MLDatabases\\data\\uci\\spamNotspam\\Youtube01-Psy.csv"
    
    with open(FILE,"r", encoding="utf8") as f:
        line=f.readline()
        for line in f:
            l=line[instr(line,",",3):len(line)-2].strip(",\"").strip("\",")

            if x_onehot_encode:
                if len(l)>max_len:
                    max_len=len(l)
                unique_chars=set(''.join(unique_chars)+l)
            x.append(l)
            y.append(int(line[-2]))
    
    FILE="C:\\MLDatabases\\data\\uci\\spamNotspam\\Youtube02-KatyPerry.csv"
    
    with open(FILE,"r", encoding="utf8") as f:
        line=f.readline()
        for line in f:
            l=line[instr(line,",",3):len(line)-2].strip(",\"").strip("\",")

            if x_onehot_encode:
                if len(l)>max_len:
                    max_len=len(l)
                unique_chars=set(''.join(unique_chars)+l)
            x.append(l)
            y.append(int(line[-2]))
    
    FILE="C:\\MLDatabases\\data\\uci\\spamNotspam\\Youtube03-LMFAO.csv"
    with open(FILE,"r", encoding="utf8") as f:
        line=f.readline()
        for line in f:
            l=line[instr(line,",",3):len(line)-2].strip(",\"").strip("\",")

            if x_onehot_encode:
                if len(l)>max_len:
                    max_len=len(l)
                unique_chars=set(''.join(unique_chars)+l)
            x.append(l)
            y.append(int(line[-2]))
    
    
    FILE="C:\\MLDatabases\\data\\uci\\spamNotspam\\Youtube04-Eminem.csv"
    with open(FILE,"r", encoding="utf8") as f:
        line=f.readline()
        for line in f:
            l=line[instr(line,",",3):len(line)-2].strip(",\"").strip("\",")

            if x_onehot_encode:
                if len(l)>max_len:
                    max_len=len(l)
                unique_chars=set(''.join(unique_chars)+l)
            x.append(l)
            y.append(int(line[-2]))
    
    FILE="C:\\MLDatabases\\data\\uci\\spamNotspam\\Youtube05-Shakira.csv"        
    with open(FILE,"r", encoding="utf8") as f:
        line=f.readline()
        for line in f:
            l=line[instr(line,",",3):len(line)-2].strip(",\"").strip("\",")

            if x_onehot_encode:
                if len(l)>max_len:
                    max_len=len(l)
                unique_chars=set(''.join(unique_chars)+l)
            x.append(l)
            y.append(int(line[-2]))
            
    FILE="C:\\MLDatabases\\data\\uci\\spamNotspam\\SMSSpamCollection"        
    with open(FILE,"r", encoding="utf8") as f:
        for line in f:            
            if line.startswith("ham"):
                if x_onehot_encode:
                    if len(line[3:None].strip())>max_len:
                        max_len=len(line[3:None].strip())
                    unique_chars=set(''.join(unique_chars)+line[3:None].strip())

                x.append(line[3:None].strip())
                y.append(1)
            else:
                if x_onehot_encode:
                    if len(line[5:None].strip())>max_len:
                        max_len=len(line[5:None].strip())  
                    unique_chars=set(''.join(unique_chars)+line[5:None].strip())
                x.append(line[5:None].strip())
                y.append(0)

    if x_onehot_encode:
        char_to_idx={chr:idx for idx,chr in enumerate(unique_chars)}
        idx_to_chr={idx:chr for idx,chr in enumerate(unique_chars)}
        for i,sen in enumerate(x):
            t=[]
            for chars in sen:
                t.append(char_to_idx[chars])
            x[i]=t

    x_train=x[0:6000]
    y_train=y[0:6000]
    x_test=x[6000:None]
    y_test=y[6000:None]
    
    print(x_train[100])
    print(  ''.join([idx_to_chr[i] for i in x_train[100]] ))
    
    return (x_train,y_train,x_test,y_test),(unique_chars,char_to_idx,idx_to_chr,max_len)
    

def plant_leaf_image_classification(do_clip):
    
    """
    https://archive.ics.uci.edu/ml/datasets/One-hundred+plant+species+leaves+data+set
    
    """
    
    
    PATH="C:\\MLDatabases\\data\\uci\\100 leaves plant\\100 leaves plant species\\data"
    dir_list=os.listdir(PATH)
    plantname_to_idx={name:idx for (idx,name) in enumerate(dir_list)}
    idx_to_plantname={idx:name for (idx,name) in enumerate(dir_list)}
    
    np.random.seed(10)
    
    labels=[]
    images=np.zeros((1600,50,50))
    
    start_ix=0
    for subfolder in dir_list:   
        imagePaths = glob.glob(PATH + '\\' + subfolder +'\\*.jpg')
        im_array = np.array( [misc.imresize(np.array(Image.open(imagePath), 'f'),(50,50)) for imagePath in imagePaths] )
        
        images[start_ix:start_ix+len(im_array)] = im_array
        start_ix += len(im_array)
        for imagePath in imagePaths:
            labels.append(plantname_to_idx[subfolder])
    
    if do_clip[0]:
        np.clip(images,do_clip[1],do_clip[2])
    
    y=np.array(labels)
    
    idx=np.linspace(0,1599,1600,dtype=np.int)
    np.random.shuffle(idx)
    np.random.shuffle(idx)
    np.random.shuffle(idx)
    
    idx_train=idx[0:1500]
    idx_test=idx[1500:None]
    
    x_train=images[idx_train]
    y_train=y[idx_train]
    
    x_test=images[idx_test]
    y_test=y[idx_test]
    
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    
    return (x_train,y_train,x_test,y_test),(plantname_to_idx,idx_to_plantname)


def plant_leat_classification_shape(do_normalize):
    
    """
    https://archive.ics.uci.edu/ml/datasets/One-hundred+plant+species+leaves+data+set
    
    """
    
    FILE="C:\\MLDatabases\\data\\uci\\100 leaves plant\\100 leaves plant species\\data_Sha_64.txt"
    
    data=np.loadtxt(FILE,dtype=">U",delimiter=",",skiprows=1,usecols=(0,))
    plantname_to_idx={name:idx for (idx,name) in enumerate(np.unique(data))}
    idx_to_plantname={idx:name for (idx,name) in enumerate(np.unique(data))}
    del data
    
    def class_converter(s):
        return plantname_to_idx[s.decode("utf-8")]
    
    data=np.loadtxt(FILE,delimiter=",",skiprows=1,converters={0:class_converter})
    
    if do_normalize:
        data=data-np.mean(data,axis=1,keepdims=True)
        data=data/np.std(data,axis=1,keepdims=True)
    
    x_train=data[0:1500,1:None]
    y_train=data[0:1500,0]
    
    x_test=data[1500:None,1:None]
    y_test=data[1500:None,0]
    
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    
    return (x_train,y_train,x_test,y_test),(plantname_to_idx,idx_to_plantname)

def plant_leat_classification_texture(do_normalize):
    
    FILE="C:\\MLDatabases\\data\\uci\\100 leaves plant\\100 leaves plant species\\data_Tex_64.txt"
    
    data=np.loadtxt(FILE,dtype=">U",delimiter=",",skiprows=1,usecols=(0,))
    plantname_to_idx={name:idx for (idx,name) in enumerate(np.unique(data))}
    idx_to_plantname={idx:name for (idx,name) in enumerate(np.unique(data))}
    del data
    
    def class_converter(s):
        return plantname_to_idx[s.decode("utf-8")]
    
    data=np.loadtxt(FILE,delimiter=",",skiprows=1,converters={0:class_converter})
    
    if do_normalize:
        data=data-np.mean(data,axis=1,keepdims=True)
        data=data/np.std(data,axis=1,keepdims=True)
    
    x_train=data[0:1500,1:None]
    y_train=data[0:1500,0]
    
    x_test=data[1500:None,1:None]
    y_test=data[1500:None,0]
    
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    
    return (x_train,y_train,x_test,y_test),(plantname_to_idx,idx_to_plantname)


def plant_leat_classification_margin(do_normalize):
    
    FILE="C:\\MLDatabases\\data\\uci\\100 leaves plant\\100 leaves plant species\\data_Mar_64.txt"
    
    data=np.loadtxt(FILE,dtype=">U",delimiter=",",skiprows=1,usecols=(0,))
    plantname_to_idx={name:idx for (idx,name) in enumerate(np.unique(data))}
    idx_to_plantname={idx:name for (idx,name) in enumerate(np.unique(data))}
    del data
    
    def class_converter(s):
        return plantname_to_idx[s.decode("utf-8")]
    
    data=np.loadtxt(FILE,delimiter=",",skiprows=1,converters={0:class_converter})
    
    if do_normalize:
        data=data-np.mean(data,axis=1,keepdims=True)
        data=data/np.std(data,axis=1,keepdims=True)
    
    x_train=data[0:1500,1:None]
    y_train=data[0:1500,0]
    
    x_test=data[1500:None,1:None]
    y_test=data[1500:None,0]
    
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    
    return (x_train,y_train,x_test,y_test),(plantname_to_idx,idx_to_plantname)
    
def truck_failure_anomaly_detection_clf(do_normalize):
    
    """
    https://archive.ics.uci.edu/ml/datasets/IDA2016Challenge
    
    """
    
    FILE="C:\\MLDatabases\\data\\uci\\truck\\to_uci\\aps_failure_training_set.csv"
    
    def class_converter(s):
        if s==b"neg":
            return 0
        else:
            return 1
        
    data=np.loadtxt(FILE,dtype=">U",delimiter=",",skiprows=1,converters={0:class_converter})
    data[data=="na"]=-1    
    data=np.asarray(data,dtype=np.float32)
    
    x_train=np.copy(data[:,1:None])
    y_train=np.copy(data[:,0])
    
    if do_normalize:
        x_train=x_train-np.mean(x_train,axis=1,keepdims=True)
        x_train=x_train/np.std(x_train,axis=1,keepdims=True)
    
    del data
    
    FILE="C:\\MLDatabases\\data\\uci\\truck\\to_uci\\aps_failure_test_set.csv"
    
    data=np.loadtxt(FILE,dtype=">U",delimiter=",",skiprows=1,converters={0:class_converter})
    data[data=="na"]=-1
    data=np.asarray(data,dtype=np.float32)
    
    x_test=data[:,1:None]
    y_test=data[:,0]
    
    if do_normalize:
        x_test=x_test-np.mean(x_test,axis=1,keepdims=True)
        x_test=x_test/np.std(x_test,axis=1,keepdims=True)
    
    print(x_train.shape,np.unique(y_train).shape,x_test.shape,y_test.shape)
    
    return (x_train,y_train,x_test,y_test)


def get_iris(do_normalize):
    
    def connv(s):
        if s.decode("utf-8")=="Iris-setosa":
            return 1
        elif s.decode("utf-8")=="Iris-versicolor":
            return 2
        else:
            return 3
    
    FILE="C:\MLDatabases\iris\iris.csv"
    
    data=np.loadtxt(FILE,delimiter=",",skiprows=0,converters={4:connv})

    if do_normalize:
        data=data-np.mean(data,axis=1,keepdims=True)
        data=data/np.std(data,axis=1,keepdims=True)        
    print(data.shape)
    return data    
        

if __name__=="__main__":
    truck_failure_anomaly_detection_clf(True)
    