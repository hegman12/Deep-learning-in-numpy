import numpy as np
import categorical_converter as cc


def fit(data):
    """Uses MSE loss    
    """
    

    np.random.seed(14)
    np.random.shuffle(data)    
    
    h=(50,50,1)
    step_size=0.1
    tolerence=0.001
    iteration_max=600
    iteration=0
    #Regularisation param, added to gradients    
    reg=0.1
    
    x_train=data[0:3500,0:-1]
    x_test=data[3500:-1,0:-1]    
    y_train=data[0:3500,-1]
    y_test=data[3500:-1,-1]
    del data
    
    K= 1#np.unique(y_train).shape[0]
    
    #x=np.loadtxt("/home/manjunath/iris/iris.csv", comments=None, delimiter=',', converters=None, usecols=(0,1,2,3))
    train_mean=np.mean(x_train,axis=1,keepdims=True)
    x_train=x_train-train_mean
    max_val=x_train.max()
    #x_train /= max_val
    #x_train /= 10
            
    #x=x-x.mean()
    
    #y=np.loadtxt("/home/manjunath/iris/iris.csv",  comments=None, delimiter=',')
    #y_train=y_to_classification_form(y_train,3)
    #y_test=y_to_classification_form(y_test,3)

    n_samples,n_features=x_train.shape
    
    
    w1=(np.random.randn(n_features*h[0]).reshape(n_features,h[0]))/np.sqrt(2/(n_features+h[0])) * 0.005
    w2=(np.random.randn(h[0]*h[1]).reshape(h[0],h[1]))/np.sqrt(2/(h[0]+h[1])) * 0.005
    w3=(np.random.randn(h[1]*h[2]).reshape(h[1],h[2]))/np.sqrt(2/(h[1]+h[2])) * 0.005
    #w3=(np.random.randn(h[1]*K).reshape(h[1],K)*0.5)/np.sqrt(2/h[1]+K)
    print(x_train.shape,w1.shape)
    #Basically no significance, added bias for completion    
    b1 = np.zeros((1,h[0]))
    b2 = np.zeros((1,h[1]))
    b3 = np.zeros((1,K))
        
    while iteration<iteration_max :
        
        #Calculate scores        
        scores_layer1=np.dot(x_train,w1)+b1 #  150 x 4
        
        #Do not use sigmoid, you will be stuck in long mess of nans and inf and overflows and div by zeros
        #x2=1/1+np.exp(-scores_layer1) # 150 x 4
        
        #Use reLU
        
        x2=np.maximum(0,scores_layer1)    
        
        score_layer2=np.dot(x2,w2)+b2
        
        x3=np.maximum(0,score_layer2)
        
        final_scores=np.dot(x3,w3)+b3 #  150 x 1
        error=y_train.reshape(3500,1)-final_scores #  150 x 1        
        
        rmse=error.mean()        
        a=np.ones(len(final_scores)).reshape(-1,1)
        #print("Iteration: ",iteration)     
        #print(final_scores)
        
        #error=np.tile(np.sum(error,axis=0,keepdims=True)/(len(x_train)),(len(x_train),))
        #Again, use softmax or sigmoid loss for classification, MSE or distance is for regression only        
        dscores=1*error
        
        #There is possibility of only 1 class for data, so use below, else the implementation will be bit complex        
        
        dw3=np.dot(x3.T,dscores)  # 4 x 1
        dx3=np.dot(w3,dscores.T)  # 4 x 150        
        dhid2=dx3.T
        dhid2[x3<=0]=0 
        
        #dprod = (x2 * (1- x2)) * dx2.T # this is wrong, find out why, we mostly need to multiply with upstream gradient       
        
        dw2=np.dot(x2.T,dhid2)  # 4 x 1
        dx2=np.dot(w2,dhid2.T)  # 4 x 150        
        
        dhid1=dx2.T
        dhid1[x2<=0]=0        
        
        #print(dprod.shape)
        
        dw1 =  np.dot( x_train.T,dhid1)

        db1=np.sum(b1,axis=0,keepdims=True)        
        db2=np.sum(b2,axis=0,keepdims=True)        
        db3=np.sum(b3,axis=0,keepdims=True)
        
        #Regularisation of gradients
        dw3 += reg*w3
        dw2 += reg*w2
        dw1 += reg*w1
        
        w1 = w1 - (step_size * dw1)
        w2 = w2 - (step_size * dw2)
        w3 = w3 - (step_size * dw3)
        b1 = b1 - (step_size * db1)
        b2 = b2 - (step_size * db2)
        b3 = b3 - (step_size * db3)
        
        if iteration%100 == 0 :
            print('Loss',rmse)
        
        iteration=iteration+1
    
    #print('FInal weights are: ', w1,w2)
    
    x_test=x_test- np.mean(x_test)
    #x_test /= max_val
    s1=np.dot(x_test,w1)
    #px2=1/1+np.exp(-s1)
    px2=np.maximum(0,s1)
        
    s2=np.dot(px2,w2)    
    px3=np.maximum(0,s2)
    
    out=np.dot(px3,w3)
    
    out=out-y_test.reshape(-1,1)
    print(out.shape)
    count=len(out[abs(out)>0])
    print(count)
    print("accuracy: ", ((676-count)/676) *100,"%")
    

if __name__ == '__main__':
    data=np.genfromtxt("/home/manjunath/Downloads/abalone.data",dtype="str",delimiter=",")
    data=cc.convert(data,(0,))
    data=np.array(data,dtype=np.float16)
    fit(data)