import numpy as np
import batch_norm as bn
import relu as relu
import loss_functions.functions as fn
import sigmoid as sigmoid
import categorical_converter2 as cc
import mnist


    
def convert_y(s):
    if s==b'Iris-setosa':
        return int(1)
    elif s==b'Iris-versicolor':
        return int(2)
    else:
        return int(3)
        
def y_to_classification_form(y,n_classes):
    
    """
    
    THIS WILL WORK, but the best solurtion is is implemented
    
    y_count=len(y)
    return_y=np.zeros(y_count*n_classes).reshape(y_count,n_classes)
    
    for ix,item in enumerate(y):        
        return_y[ix,int(item)-1]=1.
    
    
    return return_y  
    """
    
    return np.eye(n_classes)[y]
    
    

def fit(x_train,x_test,y_train,y_test):
    """temp for testing, load data from locafolder    
    """
    #data=np.loadtxt("/home/manjunath/iris/iris.csv", comments=None, delimiter=',', usecols=(0,1,2,3,4), converters={4: convert_y })  

    h=(10,10,10)
    step_size=0.001
    tolerence=0.001
    iteration_max=1000
    iteration=0
    #Regularisation param, added to gradients    
    reg=0.01
    
    K=np.unique(y_train).shape[0]
    
    #x=np.loadtxt("/home/manjunath/iris/iris.csv", comments=None, delimiter=',', converters=None, usecols=(0,1,2,3))
   
    
    """
    
    train_mean=np.mean(x_train,axis=0)
    x_train=x_train-train_mean
    #std_x = np.sqrt(np.sum(np.square(x_train - train_mean),axis=0)/x_train.shape[1])
    std_x=np.std(x_train,axis=0)
    x_train=x_train/std_x
    
    x_test=x_test - train_mean
    x_test=x_test/std_x

    """



    y_train=y_to_classification_form(y_train,K)
    y_test=y_to_classification_form(y_test,K)

    n_samples,n_features=x_train.shape
    gamma2=np.random.randn(h[0]).reshape(1,h[0])
    beta2=np.random.randn(h[0]).reshape(1,h[0])
    gamma3=np.random.randn(h[1]).reshape(1,h[1])
    beta3=np.random.randn(h[1]).reshape(1,h[1])
    eps=0.001
    
    w1=(np.random.randn(n_features*h[0]).reshape(n_features,h[0]))/np.sqrt(2/(n_features+h[0]))
    w2=(np.random.randn(h[0]*h[1]).reshape(h[0],h[1]))/np.sqrt(2/(h[0]+h[1]))
    w3=(np.random.randn(h[1]*h[2]).reshape(h[1],h[2]))/np.sqrt(2/(h[1]+h[2]))
    
    dw1_priv=np.zeros(w1.shape)
    dw2_priv=np.zeros(w2.shape)
    dw3_priv=np.zeros(w3.shape)
    
    #w3=(np.random.randn(h[1]*K).reshape(h[1],K)*0.5)/np.sqrt(2/h[1]+K)
    #Basically no significance, added bias for completion
    b1 = np.zeros((1,h[0]))
    b2 = np.zeros((1,h[1]))
    b3 = np.zeros((1,K))
        
    while iteration<iteration_max :
        
        #Calculate scores        
        scores_layer1=np.dot(x_train,w1)+b1 #  125x4,4x10 = 125x10
        #print("iteration",iteration, "first layer",np.any(np.isnan(scores_layer1)))
        #Do not use sigmoid, you will be stuck in long mess of nans and inf and overflows and div by zeros
        #x2=1/1+np.exp(-scores_layer1) # 150 x 4
        
        #Use reLU
        
        #x2=np.maximum(0,scores_layer1)
        bn_x2,bn_cache2=bn.batch_norm_forword(scores_layer1,gamma2,beta2) #125x10
        #print("iteration",iteration, "first layer BN",np.any(np.isnan(bn_x2)))
        #x2=relu.relu_forword(bn_x2.T)
        x2=relu.relu_forword(bn_x2) #125x10
        #print("iteration",iteration, "first layer relu",np.any(np.isnan(x2)))
        
        score_layer2=np.dot(x2,w2)+b2  #125x10,10x10=125x10
        #print("iteration",iteration, "second layer",np.any(np.isnan(score_layer2)))
        bn_x3,bn_cache3=bn.batch_norm_forword(score_layer2,gamma3,beta3) #125x10
        x3=relu.relu_forword(bn_x3) #125x10 
        
        final_scores=np.dot(x3,w3)+b3 #  125x10,10x3=125x3
        
        #Again, use softmax or sigmoid loss for classification, MSE or distance is for regression only        
        
        probs=fn.softmax(final_scores) #125x3
        
         
        
        dscores=fn.cross_enropy_grad_singleclass(probs,y_train) #  125x3
        #There is possibility of only 1 class for data, so use below, else the implementation will be bit complex        
        #print(x3.shape)
        dw3=np.dot(x3.T,dscores)  # 10x125,125x3=10x3
        dx3=np.dot(w3,dscores.T)  # 10x3,3x125=10x125
        
        #dhid2=dx3.T
        #dhid2[x3<=0]=0
        
        dhid2=relu.relu_backword(dx3.T,x3) #125x10
        #print("dhid2",dhid2.shape)
        bn_dhid2,dgamma3,dbeta3=bn.batch_norm_backword(dhid2,bn_cache3) #125x10
        #dprod = (x2 * (1- x2)) * dx2.T # this is wrong, find out why, we mostly need to multiply with upstream gradient       
        
        dw2=np.dot(x2.T,bn_dhid2)  # 10x125,125x10=10x10
        dx2=np.dot(w2,dhid2.T)  #10x10,10x125=10x125
        
        #dhid1=dx2.T
        #dhid1[x2<=0]=0
        
        dhid1=relu.relu_backword(dx2.T,x2) #125x10
        
        bn_dx2,dgamma2,dbeta2=bn.batch_norm_backword(dhid1,bn_cache2) #125x10
        #print(dprod.shape)
        
        dw1 =  np.dot( x_train.T,bn_dx2) #  125x4,12510=4x10

        db1=np.sum(b1,axis=0,keepdims=True)        
        db2=np.sum(b2,axis=0,keepdims=True)        
        db3=np.sum(b3,axis=0,keepdims=True)
        
        #Regularisation of gradients
        
        #Optimisation
        
        #dw1 = (dw1+dw1_priv)/2
        #dw2 = (dw2+dw2_priv)/2
        #dw3 = (dw3+dw3_priv)/2
        
        dw3 += reg*w3
        dw2 += reg*w2
        dw1 += reg*w1
        
        w1 = w1 - (step_size * dw1)
        w2 = w2 - (step_size * dw2)
        w3 = w3 - (step_size * dw3)
        
        #print(dw1)
        #print(dw2)
        #print(dw3)
        
        #dw1_priv=dw1
        #dw2_priv=dw2
        #dw3_priv=dw3
        
        """
        redundant parameters after batch normalization                
        """
        
        b1 = b1 - (step_size * db1)
        b2 = b2 - (step_size * db2)
        b3 = b3 - (step_size * db3)
        

        
        gamma2= gamma2 - (step_size * dgamma2)
        beta2 = beta2 - (step_size * dbeta2)
        gamma3= gamma3 - (step_size * dgamma3)
        beta3 = beta3 - (step_size * dbeta3)

        
        if iteration%10 == 0 :
            #print("****iteration:",iteration)
            #x_test /= 10          
            
            s1=np.dot(x_test,w1)
            #px2=1/1+np.exp(-s1)
            bn_x2t,bn_cache2t=bn.batch_norm_forword(s1,gamma2,beta2)
            px2=relu.relu_forword(bn_x2t)
                
            s2=np.dot(px2,w2)    
            bn_x3t,bn_cache3t=bn.batch_norm_forword(s2,gamma3,beta3)
            px3=relu.relu_forword(bn_x3t)
            
            out=np.dot(px3,w3)
            
            counter=0
            for y_p,y_a in zip(np.argmax(out,axis=1),y_test):
                if np.argmax(y_a)==y_p:
                    counter +=1
            print("accuracy: ", (counter/10000) *100,"%")
            loss=fn.cross_entropy_loss_singleclass(probs,y_train) # scalar
            print('Loss',loss/n_samples)
            
            dw1_p=np.zeros_like(dw1)
            dw2_p=np.zeros_like(dw2)
            dw3_p=np.zeros_like(dw3)
            
            print("dw1",dw1==dw1_p)
            print("dw1",dw2==dw2_p)
            print("dw1",dw3==dw3_p)
            
            dw1_p=dw1
            dw2_p=dw2
            dw3_p=dw3
            
            #print("gamma2",gamma2)
            #print("beta2",beta2)
        
        iteration=iteration+1
    
    #print('FInal weights are: ', w1,w2)
    
    

if __name__ == '__main__':
    """
    data=np.genfromtxt("/home/manjunath/Downloads/abalone.data",dtype="str",delimiter=",")
    data=cc.convert(data,(0,))
    data=np.array(data,dtype=np.float16)
    
    np.random.seed(14)
    np.random.shuffle(data)
    
    x_train=data[0:3500,0:-1]
    x_test=data[3500:-1,0:-1]    
    y_train=data[0:3500,-1]
    y_test=data[3500:-1,-1]
    del data
    """
    
    """
    data=np.loadtxt("C:\\MLDatabases\\iris\\iris.csv", comments=None, delimiter=',', usecols=(0,1,2,3,4), converters={4: convert_y })
    np.random.seed(14)
    np.random.shuffle(data)
    x_train=data[0:125,0:-1]
    x_test=data[125:-1,0:-1]    
    y_train=data[0:125,-1]
    y_test=data[125:-1,-1]
    del data
    
    """
    
    """
    
    data=np.genfromtxt("C:\\MLDatabases\\iris\\bank-full.csv",dtype="str",delimiter=";")
    data=np.delete(data,0,0)
    data=cc.convert(data,(1,2,3,4,6,7,8,10,15,16))
    data=np.array(data,dtype=np.float16)
    
    data=data[~np.isnan(data).any(axis=1)]
    data=data[~np.isinf(data).any(axis=1)]
    print(data.shape)
    if np.sum(np.isinf(data))>0 or np.sum(np.isnan(data))>0 :
        print("has invalid values") 
    
    np.random.seed(14)
    np.random.shuffle(data)
    
    x_train=data[0:40000,0:-1]
    x_test=data[40000:-1,0:-1]    
    y_train=data[0:40000,-1]
    y_test=data[40000:-1,-1]
    del data    
    
    """
   
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()

    test_images = mnist.test_images()
    test_labels = mnist.test_labels()
    
    print("train_images",train_images.shape)
    print("test_images",test_images.shape)
    print("test_labels",test_labels.shape)
    print("train_labels",test_labels.shape)
    
    #fit(x_train,x_test,y_train,y_test)
    fit(train_images.reshape(60000,28*28),test_images.reshape(10000,28*28),train_labels,test_labels)
    