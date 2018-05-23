import tensorflow as tf
import matplotlib.pyplot as plt
import datasets
import numpy as np

tf.set_random_seed(100)
tf.reset_default_graph()
# Get current size
fig_size = plt.rcParams["figure.figsize"]

# Prints: [8.0, 6.0]
print ("Current size:", fig_size)
 
# Set figure width to 12 and height to 9
fig_size[0] = 8
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size

def graph():
    x=tf.placeholder(tf.float32,(None,178),"x")
    y=tf.placeholder(tf.int64,(None,),"y")
    w=tf.Variable(tf.truncated_normal(shape=(178,2),mean=0,stddev=0.3),name="w")
    b=tf.Variable(tf.truncated_normal(shape=(),mean=0,stddev=1),name="b")
    h=tf.add(tf.matmul(x,w),b,name="out")
    logit=tf.nn.relu(h)
    op=tf.nn.softmax(logit,dim=1)
    loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=logit)
    opt=tf.train.AdamOptimizer(0.01)
    grad_var=opt.compute_gradients(loss)
    grad_var=[(tf.clip_by_value(g,-5000, +5000, name=None),v) for g,v in grad_var]
    apply_grad=opt.apply_gradients(grad_var)
    accu=tf.reduce_mean(tf.cast(tf.equal(y,tf.argmax(logit,axis=1)),tf.float32))
    #yy=tf.reduce_mean(y)
    #oop=tf.reduce_mean(op)
    #accu=tf.reduce_mean(tf.cast(tf.cond(tf.logical_and((yy-oop)<1,(yy-oop)> -1) ,lambda:True,lambda:False),tf.float32))
    
    return x,y,op,loss,accu,apply_grad,grad_var

x,y,op,loss,accu,apply_grad,grad_var=graph()
saver=tf.train.Saver()

#x_train,y_train,x_test,y_test=datasets.power_plant_data_regression(True)

#data,p_index=datasets.truck_failure_anomaly_detection_clf(True)
x_train,y_train,x_test,y_test=datasets.epileptic_EEG_classification(True)

sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
for i in range(300):
    o,l,a,_,gv=sess.run([op,loss,accu,apply_grad,grad_var],feed_dict={x:x_train,y:y_train})
    if i%1==0:
        print("Iteration",i,"Loss",np.mean(l),"acc",a)
 
    if i%299==0:
        o,l,a=sess.run([op,loss,accu],feed_dict={x:x_test,y:y_test})
        print("Iteration",i,"Loss",np.mean(l),"Accuracy",a)
        #print(np.unique(o.ravel()))
        
        plt.scatter(x_test[:,1],x_test[:,2],c=y_test, s=10, cmap=plt.cm.Spectral)
        #plt.xticks(np.arange(0, 16000, step=1000))
        #plt.yticks(np.arange(0, 170, step=10))
        plt.show()
        

saver.save(sess,"D:\\DL\\tf_relu_classification")
sess.close()
