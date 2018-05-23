import numpy as np
np.random.seed(90)
def load_data(path):
    sentenses,labels=[],[]
    with open(path) as file:        
        for line in file:
            sentenses.append(line.split("\t")[0].strip())
            labels.append(int(line.split("\t")[1].strip()))
    return sentenses,labels

sentenses,labels=load_data("C:\\MLDatabases\\data\\rnn\\yelp_labelled.txt")
"""
sentenses1,labels1 = load_data("C:\\MLDatabases\\data\\rnn\\imdb_labelled.txt")
sentenses += sentenses1
labels += labels1
sentenses1,labels1 = load_data("C:\\MLDatabases\\data\\rnn\\amazon_cells_labelled.txt")
sentenses += sentenses1
labels += labels1

del labels1
del sentenses1
"""
x=[]

max_sent_len=max([len(s.split()) for s in sentenses])

print("largest sentense has ",str(max_sent_len)," words")

sentenses = [sentens.split() for sentens in sentenses]
words=[]

for sentense in sentenses:
    for word in sentense:
        words.append(word)
words.append('0')
word_to_index={w:i for i,w in enumerate(set(words))}

for i,sentense in enumerate(sentenses):
    wtemp=[]
    for j,word in enumerate(sentense):
        wtemp.append(word_to_index[word])
    x.append(wtemp)
del wtemp
x_eye=np.eye(len(set(words)))
voc_size=len(set(words))

x_size=len(sentenses)
del words
del sentenses

y_onehot=np.eye(2)[np.array(labels)]
print("y onehot ",y_onehot.shape)
x=np.array(x)

hid_size=100
out_dim=2

wu=np.random.randn(hid_size,voc_size)
ww=np.random.randn(hid_size,hid_size)
wv=np.random.randn(out_dim,hid_size)
s=np.zeros((hid_size,1))

dwu=np.zeros_like(wu)
dww=np.zeros_like(ww)
dwv=np.zeros_like(wv)
ds=np.zeros_like(s)

mwu=np.zeros_like(wu)
mww=np.zeros_like(ww)
mwv=np.zeros_like(wv)

dmwu=np.zeros_like(wu)
dmww=np.zeros_like(ww)
dmwv=np.zeros_like(wv)

lr=0.001

x_train=np.array(x[0:800])
y_train=y_onehot[0:800]

x_test=np.array(x[800:])
y_test=y_onehot[800:]

for epoch in range(100):
    loss=0
    for i,sen in enumerate(x_train):
        sen=np.hstack((sen,np.zeros((71-len(sen)))))
        xs_onehot=x_eye[sen.astype(int)]
        
        hs={}
        hs[-1]=np.copy(s)
        
        for j,word in enumerate(xs_onehot):
            j_track=j
            inter1=np.dot(wu,word.reshape(voc_size,1))
            inter2=np.dot(ww,hs[j-1])
            inter3=inter1   +   inter2
            hs[j]= np.tanh(inter3)
        
        s=np.copy(hs[j_track])
        out = np.dot(np.array(hs[j_track]).T,wv.T)    
        out_prob= np.exp(out)/np.sum(np.exp(out))
        out_prob=out_prob[0]
        
        loss = -np.log(out_prob[int(np.argmax(y_train[i]))])
        
        print("loss ", loss)
        
        dout=out_prob
        dout[np.argmax(y_train[i])] -= 1
        
        dwv=np.dot(hs[j_track],dout.reshape(1,2)).reshape(out_dim,hid_size)
        dlast_hid=np.dot(wv.T,dout.reshape(2,1)) + ds
        
        for j in reversed(range(len(xs_onehot))):
            dinter3=(1.-hs[j]*hs[j]) * dlast_hid
            dinter1=dinter3*1
            dinter2=dinter3*1
            
            dwu += np.dot(dinter1,xs_onehot[j].reshape(1,voc_size))
            dww += np.dot(hs[j-1],inter2.T)
            
            """
            if len(xs_onehot)-1 == j:
                ds=np.dot(ww,inter2)
            """
        np.clip(dwv,-5,5,out=dwv)
        np.clip(dww,-5,5,out=dww)
        np.clip(dwu,-5,5,out=dwu)
        np.clip(ds,-5,5,out=ds)
        
        mwv += dwv*dwv
        mwu += dwu*dwu
        mww += dww*dww
        
        wv += -lr*dwv/np.sqrt(mwv+ 1e-8)
        wu += -lr*dwu/np.sqrt(mwu+1e-8)
        ww += -lr*dww/np.sqrt(mww+1e-8)
        
        dwu=np.zeros_like(wu)
        dww=np.zeros_like(ww)
        dwv=np.zeros_like(wv)
    
    print("Loss at epoch",epoch,"  :",loss)
    
    if epoch%10==0:
        for n,sen in enumerate(x_test):
            
            test_track=[]
            
            sen=np.hstack((sen,np.zeros((71-len(sen)))))
            xs_onehot=x_eye[sen.astype(int)]
            
            hs={}
            hs[-1]=np.copy(s)
            
            for l,word in enumerate(xs_onehot):
                j_track=l
                inter1=np.dot(wu,word.reshape(voc_size,1))
                inter2=np.dot(ww,hs[l-1])
                inter3=inter1   +   inter2
                hs[l]= np.tanh(inter3)
            out = np.dot(np.array(hs[j_track]).T,wv.T)    
            out_prob= np.exp(out)/np.sum(np.exp(out))
            out_prob=out_prob[0]
            test_track.append(np.argmax(out_prob))
        
        print("Accuracy at epoch ",epoch, (np.sum(np.argmax(y_test,axis=1)==np.array(test_track))/200) * 100)
            


    
    
    















