
# coding: utf-8

# In[ ]:

import sys


# In[ ]:

import numpy as np


# #train

# In[ ]:

X = []
y = []


# In[ ]:

with open('/u/cs246/data/adult/a7a.train','r') as f:
    for line in f:
        entries = line.strip().split(' ')
        y.append((int)(entries[0]))
        temp_X = np.zeros(124)
        for entry in entries[1:]:
            index,value = map(int, entry.split(':'))
            temp_X[index] = value
        X.append(temp_X)


# In[ ]:

w = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(X),X)),np.transpose(X)),y)


# In[ ]:

w = w.reshape(1,124)


# In[ ]:

def predict(X,w,y):
    y_predict = np.dot(w,np.transpose(X))
    y_predict = np.copysign(1,y_predict)
    x2 = np.size(y_predict)
    x1 = np.size(y_predict)-np.count_nonzero(y-y_predict)
    x3 = float(x1)/x2
    return(x1,x2,x3)


# In[ ]:

(x1,x2,x3)=predict(X,w,y)


# In[ ]:

print('train dataset')
print(x1,' correct predictions for',x2,'points')
print('The accuracy is',x3)


# #dev

# In[ ]:

X1 = []
y1 = []


# In[ ]:

with open('/u/cs246/data/adult/a7a.dev','r') as f1:
    for line in f1:
        entries = line.strip().split(' ')
        y1.append((int)(entries[0]))
        temp_X1 = np.zeros(124)
        for entry in entries[1:]:
            index,value = map(int, entry.split(':'))
            temp_X1[index] = value
        X1.append(temp_X1)


# In[ ]:

l = 0 #regularization coefficient


# In[ ]:

w1 = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(X1),X1)+l*np.eye(124)),np.transpose(X1)),y1)


# In[ ]:

np.shape(np.dot(np.transpose(X1),X1))


# In[ ]:

w1 = w1.reshape(1,124)


# In[ ]:

(x1,x2,x3)=predict(X1,w1,y1)


# In[ ]:

print('dev dataset')
print(x1,'correct predictions for',x2,'points')
print('The accuracy is',x3)


# #test

# In[ ]:

X2 = []
y2 = []


# In[ ]:

with open('/u/cs246/data/adult/a7a.test','r') as f2:
    for line in f2:
        entries = line.strip().split(' ')
        y2.append((int)(entries[0]))
        temp_X2 = np.zeros(124)
        for entry in entries[1:]:
            index,value = map(int, entry.split(':'))
            temp_X2[index] = value
        X2.append(temp_X2)


# In[ ]:

(x1,x2,x3)=predict(X2,w1,y2)


# In[ ]:

print('test dataset')
print(x1,'correct predictions for',x2,'points')
print('The accuracy is',x3)


# #extra test

# In[ ]:

X3 = []
y3 = []


# In[ ]:

with open(sys.argv[1],'r') as f3:
    for line in f3:
        entries = line.strip().split(' ')
        y3.append((int)(entries[0]))
        temp_X3 = np.zeros(124)
        for entry in entries[1:]:
            index,value = map(int, entry.split(':'))
            temp_X3[index] = value
        X3.append(temp_X3)


# In[ ]:

(x1,x2,x3)=predict(X3,w1,y3)


# In[ ]:

print ('test dataset')
print(x1,'correct predictions for',x2,'points')
print('The accuracy is',x3)


# In[ ]:



