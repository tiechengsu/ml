
# coding: utf-8

# In[18]:

import numpy as np


# In[19]:

X = []
y = []
X1 = []
y1 = []
X2 = []
y2 = []


# In[20]:

with open('/u/cs246/data/adult/a7a.train','r') as f:
    for line in f:
        entries = line.strip().split(' ')
        y.append((int)(entries[0]))
        temp_X = np.zeros(123)
        for entry in entries[1:]:
            index,value = map(int, entry.split(':'))
            temp_X[index-1] = value
        X.append(temp_X)


# In[21]:

with open('/u/cs246/data/adult/a7a.dev','r') as f1:
    for line in f1:
        entries = line.strip().split(' ')
        y1.append((int)(entries[0]))
        temp_X = np.zeros(123)
        for entry in entries[1:]:
            index,value = map(int, entry.split(':'))
            temp_X[index-1] = value
        X1.append(temp_X)


# In[22]:

with open('/u/cs246/data/adult/a7a.test','r') as f2:
    for line in f2:
        entries = line.strip().split(' ')
        y2.append((int)(entries[0]))
        temp_X = np.zeros(123)
        for entry in entries[1:]:
            index,value = map(int, entry.split(':'))
            temp_X[index-1] = value
        X2.append(temp_X)


# In[23]:

y = np.asanyarray(y)
X = np.asanyarray(X)
y1 = np.asanyarray(y1)
X1 = np.asanyarray(X1)
y2 = np.asanyarray(y2)
X2 = np.asanyarray(X2)


# In[24]:

y = y.reshape(np.size(y),1)
y1 = y1.reshape(np.size(y1),1)
y2 = y2.reshape(np.size(y2),1)


# In[25]:

def Count(X,y):                                         #count the number of train set c(x|y) 
    count = np.zeros((4,len(X[0])))
    N = np.sum(y[np.where(y==1)])                       #c(y=1)
    count[0,:] = np.sum(X[np.where(y==1)[0],:],axis=0)  #c(x=1|y=1)
    count[1,:] = N - count[0,:]                         #c(x=0|y=1)
    count[2,:] = np.sum(X[np.where(y==-1)[0],:],axis=0) #c(x=1|y=-1)
    count[3,:] = np.size(y)-N-count[2,:]                #c(x=0|y=-1)
    return(count,N)


# In[26]:

(count,N) = Count(X,y)


# In[27]:

def calculate(N,count,X,y1,y,alpha):
    P_xy = np.zeros((4,123))
    P_xy[0:2,:] = (count[0:2,:] + alpha)/float(N + 2*alpha)                             #P(x|y) with Dirichlet smoothing
    P_xy[2:4,:] = (count[2:4,:] + alpha)/float(np.size(y) - N + 2*alpha)
    P_temp = np.zeros((len(X),len(X[0]),2))
    P_yx = np.zeros((len(X),2))
    y_predict = np.zeros((len(X),1))
    P_temp[:,:,0] = X[:,:]*P_xy[0,:]+(-X[:,:]+1)*P_xy[1,:]                              #P(x|y=1)
    P_temp[:,:,1] = X[:,:]*P_xy[2,:]+(-X[:,:]+1)*P_xy[3,:]                              #P(x|y=0)
    P_yx=np.prod((P_temp),axis=1)*np.array([float(N)/np.size(y),1-float(N)/np.size(y)]) #product of P(x|y)
    for i in range(np.size(y1)):
        if P_yx[i,0]>P_yx[i,1]:
            y_predict[i] = 1
        else:
            y_predict[i] = -1
    return(y_predict)


# In[56]:

def classification_error(N,count,X1,y1,y):
    error = np.zeros((500,1))
    alpha = np.linspace(0,499,500)
    for i in range(500):
        y1_predict = calculate(N,count,X1,y1,y,alpha[i])
        error[i] =np.count_nonzero(y1-y1_predict)/float(np.size(y1))
    return(error)


# In[57]:

error = classification_error(N,count,X1,y1,y)


# In[58]:

np.argmin(error)


# In[59]:

y1_predict = calculate(N,count,X1,y1,y,308)


# In[61]:

print('Classification error of the dev set',np.count_nonzero(y1-y1_predict)/float(np.size(y1)))


# import matplotlib.pylab as plt

# plt.plot(error)
# plt.xlabel('alpha')
# plt.ylabel('classification error')
# plt.title('dev set')

# plt.show()

# In[67]:

y2_predict = calculate(N,count,X2,y2,y,308)


# In[68]:

print('Classification error of the test set',np.count_nonzero(y2-y2_predict)/float(np.size(y2)))


# In[ ]:



