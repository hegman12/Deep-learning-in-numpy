import tensorflow as tf
import matplotlib.pyplot as plt
import datasets
import numpy as np

tf.set_random_seed(100)
tf.reset_default_graph()
plt.ion()

# Get current size
fig_size = plt.rcParams["figure.figsize"]
 
# Prints: [8.0, 6.0]
print ("Current size:", fig_size)
 
# Set figure width to 12 and height to 9
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size

def graph():
    x=tf.placeholder(tf.float32,(None,4),"x")
    y=tf.placeholder(tf.float32,(None,),"y")
    w=tf.Variable(tf.truncated_normal(shape=(4,1),mean=0,stddev=1),name="w")
    b=tf.Variable(tf.truncated_normal(shape=(),mean=0,stddev=1),name="b")
    op=tf.add(tf.matmul(x,w),b,name="out")
    loss=0.5*tf.reduce_mean(tf.square(y-op))
    opt=tf.train.GradientDescentOptimizer(0.007)
    grad_var=opt.compute_gradients(loss)
    grad_var=[(tf.clip_by_value(g,-5000, +5000, name=None),v) for g,v in grad_var]
    apply_grad=opt.apply_gradients(grad_var)
    accu=tf.reduce_mean(tf.cast(tf.equal(y,op),tf.float32))
    #yy=tf.reduce_mean(y)
    #oop=tf.reduce_mean(op)
    #accu=tf.reduce_mean(tf.cast(tf.cond(tf.logical_and((yy-oop)<1,(yy-oop)> -1) ,lambda:True,lambda:False),tf.float32))
    
    return x,y,op,loss,accu,apply_grad,grad_var

x,y,op,loss,accu,apply_grad,grad_var=graph()
saver=tf.train.Saver()

x_train,y_train,x_test,y_test=datasets.power_plant_data_regression(True)

sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)
for i in range(300):
    o,l,a,_,gv=sess.run([op,loss,accu,apply_grad,grad_var],feed_dict={x:x_train,y:y_train})
    if i%10==0:
        print("Iteration",i,"Loss",l,"acc",a)
        
    
    if i%50==0:
        o,l,a=sess.run([op,loss,accu],feed_dict={x:x_test,y:y_test})
        print("Iteration",i,"Loss",l,"Accuracy",a)
        #print(o.ravel().shape,y.shape)
        plt.plot(y_test,'b.-',o.ravel(),'r.-')
        plt.xticks(np.arange(0, 1600, step=100))
        plt.yticks(np.arange(0, 600, step=50))
        plt.show()
        

saver.save(sess,"D:\\DL\\tf_test")
sess.close()
