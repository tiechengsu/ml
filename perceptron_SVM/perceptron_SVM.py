
# coding: utf-8

# In[151]:

import numpy as np


# In[152]:

X = []
y = []
X1 = []
y1 = []
X2 = []
y2 = []


# In[153]:

#with open('E:/2016Spring/ML/hw2/a7a.train','r') as f:
with open('/u/cs246/data/adult/a7a.train','r') as f:
    for line in f:
        entries = line.strip().split(' ')
        y.append((int)(entries[0]))
        temp_X = np.zeros(123)
        for entry in entries[1:]:
            index,value = map(int, entry.split(':'))
            temp_X[index-1] = value
        X.append(temp_X)


# In[154]:

#with open('E:/2016Spring/ML/hw2/a7a.dev','r') as f1:
with open('/u/cs246/data/adult/a7a.dev','r') as f1:
    for line in f1:
        entries = line.strip().split(' ')
        y1.append((int)(entries[0]))
        temp_X = np.zeros(123)
        for entry in entries[1:]:
            index,value = map(int, entry.split(':'))
            temp_X[index-1] = value
        X1.append(temp_X)


# In[155]:

#with open('E:/2016Spring/ML/hw2/a7a.test','r') as f2:
with open('/u/cs246/data/adult/a7a.test','r') as f2:
    for line in f2:
        entries = line.strip().split(' ')
        y2.append((int)(entries[0]))
        temp_X = np.zeros(123)
        for entry in entries[1:]:
            index,value = map(int, entry.split(':'))
            temp_X[index-1] = value
        X2.append(temp_X)


# In[156]:

y = np.asanyarray(y)
X = np.asanyarray(X)
y1 = np.asanyarray(y1)
X1 = np.asanyarray(X1)
y2 = np.asanyarray(y2)
X2 = np.asanyarray(X2)


# In[157]:

y = y.reshape(np.size(y),1)
y1 = y1.reshape(np.size(y1),1)
y2 = y2.reshape(np.size(y2),1)


# In[158]:

[N,cols]=np.shape(X)
[N1,cols]=np.shape(X1)
[N2,cols]=np.shape(X2)


# #perceptron

# In[159]:

X_append = np.concatenate((X,np.ones((N,1))),axis=1)
X1_append = np.concatenate((X1,np.ones((N1,1))),axis=1)
X2_append = np.concatenate((X2,np.ones((N2,1))),axis=1)


# In[160]:

w = np.zeros((cols+1,1))


# In[161]:

eta = 0.2


# In[162]:

def perceptron(w,eta,N,X_append,y):
    for i in range(N):
        a = np.dot(X_append[i,:],w)
        t = float((np.exp(a)-np.exp(-a)))/(np.exp(a)+np.exp(-a))
        E = (t-y[i,0])*(1+t)*(1-t)*X_append[i,:]
        E = E.reshape(np.size(E),1)
        w = w - eta*E
    return(w)


# In[163]:

def predict(X,w,y):
    y_predict = np.dot(X,w)
    y_predict = np.copysign(1,y_predict)
    x2 = np.size(y_predict)
    x1 = np.size(y_predict)-np.count_nonzero(y-y_predict)
    x3 = float(x1)/x2
    return(x1,x2,x3)


# In[164]:

for i in range(70):
    w = perceptron(w,eta,N1,X1_append,y1)
    (x1,x2,x3)=predict(X_append,w,y)
    x3_max = 0
    if x3>x3_max:
        x1_max=x1
        x3_max = x3
        w_best = w


# In[165]:

print('train dataset')
print(x1_max,' correct predictions for',x2,'points')
print('The accuracy is',x3_max)


# In[166]:

(x1,x2,x3)=predict(X2_append,w_best,y2)


# In[167]:

print('test dataset')
print(x1,' correct predictions for',x2,'points')
print('The accuracy is',x3)


# #SVM

# In[168]:

w = np.zeros((cols,1))


# In[169]:

def svm(X,w,y,eta,C,b,N):
    for i in range(N):
        if 1-y[i,0]*(np.dot(X[i,:],w)+b)>0:
            w = w - eta*(float((1/N))*w-C*y[i,0]*X[i,:].reshape(np.size(X[i,:]),1))
            b = b + eta*C*y[i,0]
        else:
            w = w -eta*(1/N)*w
    return(w,b)


# In[170]:

def predict1(X,w,y,b):
    y_predict = np.dot(X,w)+b
    y_predict = np.copysign(1,y_predict)
    x2 = np.size(y_predict)
    x1 = np.size(y_predict)-np.count_nonzero(y-y_predict)
    x3 = float(x1)/x2
    return(x1,x2,x3)


# In[171]:

def performation_C(X1,y1,X,w,y,eta):
    accuracy = np.zeros((50,1))
    C = np.linspace(0,1,50)
    x1_max = 0
    for i in range(50):
        for j in range(20):
            (w,b) = svm(X1,w,y1,eta,C[i],0,N1)
            (x1,x2,accuracy[i])=predict1(X,w,y,b)
            if x1>x1_max:
                x1_max=x1
                w_best = w
                b_best = b
                C_best = C[i]
    return(C,C_best,accuracy,x1_max,w_best,b_best)


# In[172]:

eta = 0.1


# In[173]:

(C,C_best,accuracy,x1_max,w_best,b_best) = performation_C(X1,y1,X,w,y,eta)


# In[174]:

print('train dataset')
print(x1_max,' correct predictions for',N,'points')
print('when C equals ',C_best,'we can obtain the highest accuracy')
print('The accuracy is',max(accuracy))


# import matplotlib.pylab as plt

# plt.plot(C,accuracy)
# plt.xlabel('C') 
# plt.ylabel('accuracy') 
# plt.title('dev set')

# plt.show()

# In[175]:

(x1,x2,x3)=predict1(X2,w_best,y2,b_best)


# In[176]:

print('dev dataset')
print(x1,' correct predictions for',N2,'points')
print('The accuracy is',x3)


# In[ ]:



