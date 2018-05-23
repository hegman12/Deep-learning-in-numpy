import numpy as np
import categorical_converter as cc
        
def y_to_classification_form(y,n_classes):
    
    y_count=len(y)
    return_y=np.zeros(y_count*n_classes).reshape(y_count,n_classes)
    
    for ix,item in enumerate(y):        
        return_y[ix,int(item)-1]=1.
    
    return return_y  


def fit():
    """temp for testing, load data from locafolder    
    """
    data=np.genfromtxt("/home/manjunath/Downloads/bank-additional/bank-additional/bank-additional.csv",dtype="str",delimiter=";")
    data=np.delete(data,10,1) #delete 10th column, last param is axis
    data=np.delete(data,0,0)
    data=cc.convert(data,(1,2,3,4,5,6,7,8,9,13,18,19))
    data=np.array(data,dtype=np.float16)
    max_val=data.max()
    data=data/max_val
    
    np.random.seed(14)
    np.random.shuffle(data)    
    
    h=(50,50,2)
    step_size=0.0001
    tolerence=0.001
    iteration_max=2000
    iteration=0
    #Regularisation param, added to gradients    
    reg=0.1
    
    x_train=data[0:3500,0:-1]
    x_test=data[3500:-1,0:-1]    
    y_train=data[0:3500,-1]
    y_test=data[3500:-1,-1]
    del data
    
    K=np.unique(y_train).shape[0]
    
    #x=np.loadtxt("/home/manjunath/iris/iris.csv", comments=None, delimiter=',', converters=None, usecols=(0,1,2,3))
    train_mean=np.mean(x_train,axis=1,keepdims=True)
    x_train=x_train-train_mean
    #x_train /= 10
            
    #x=x-x.mean()
    
    #y=np.loadtxt("/home/manjunath/iris/iris.csv",  comments=None, delimiter=',')
    y_train=y_to_classification_form(y_train,K)
    y_test=y_to_classification_form(y_test,K)

    n_samples,n_features=x_train.shape
    
    
    w1=(np.random.randn(n_features*h[0]).reshape(n_features,h[0]))/np.sqrt(2/(n_features+h[0]))*0.05
    w2=(np.random.randn(h[0]*h[1]).reshape(h[0],h[1]))/np.sqrt(2/(h[0]+h[1])) * 0.05
    w3=(np.random.randn(h[1]*h[2]).reshape(h[1],h[2]))/np.sqrt(2/(h[1]+h[2])) * 0.05
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
        
        #error=y_train-final_scores #  150 x 1
        
        
        #error=np.tile(np.sum(error,axis=0,keepdims=True)/(len(x_train)),(len(x_train),))
        #Again, use softmax or sigmoid loss for classification, MSE or distance is for regression only        

        exp_scores=np.exp(final_scores)
        sum_exp_scores=np.sum(exp_scores,axis=1,keepdims=True)
        
        probs=exp_scores/sum_exp_scores        
        dscores=probs-y_train #  150 x 1
        
        loss=0
        for i in range(n_samples):
            loss += -np.log(probs[i,np.argmax(y_train[i])])
        
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
            print('Loss',loss/n_samples)
        
        iteration=iteration+1
    
    #print('FInal weights are: ', w1,w2)
    
    x_test=x_test- np.mean(x_test)
    x_test /= max_val
    #x_test /= 10
    s1=np.dot(x_test,w1)
    #px2=1/1+np.exp(-s1)
    px2=np.maximum(0,s1)
        
    s2=np.dot(px2,w2)    
    px3=np.maximum(0,s2)
    
    out=np.dot(px3,w3)
    
    counter=0
    for y_p,y_a in zip(np.argmax(out,axis=1),y_test):
        if np.argmax(y_a)==y_p:
            counter +=1
    print("accuracy: ", (counter/620) *100,"%")
    

if __name__=="__main__":
    fit()