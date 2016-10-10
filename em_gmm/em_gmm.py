
# coding: utf-8

# In[21]:

import numpy as np
import math
import matplotlib.pylab as plt
import pdb
get_ipython().magic('pylab inline')


# In[22]:

x1 = []
x2 = []
#with open('E:/2016Spring/ML/hw7/points.dat.txt') as f:
with open('points.dat.txt') as f:
    for line in f:
        entries = line.strip().split(' ')
        x1.append(entries[0])
        x2.append(entries[-1])
x = [x1,x2]
x = np.transpose(np.asarray(x,dtype='float64'))


# In[23]:

#use the final 1/10 of data for dev
x_train = x[:np.shape(x)[0]*0.9,:]
x_dev = x[np.shape(x)[0]*0.9:,:]


# In[24]:

def gaussian(x,mu,sigma):
    g = np.exp(-0.5*np.dot(np.dot(np.transpose(x-mu),np.linalg.pinv(sigma)),(x-mu)))/(2*math.pi*math.sqrt(np.linalg.det(sigma)))
    return(g)


# In[25]:

def new_alpha(p_z_x,k,N):
    return(np.sum(p_z_x[:,k])/N)


# In[26]:

def new_mu(x,p_z_x,k,N):
    mu_k = np.zeros((1,2),dtype='float64')
    for i in range(N):
        mu_k += p_z_x[i][k]*x[i]
    return mu_k/np.sum(p_z_x[:,k])


# In[27]:

def new_sigma(x,p_z_x,k,N,mu):
    sigma_k = np.zeros((2,2),dtype='float64')
    for i in range(N):
        sigma_k += p_z_x[i][k]*np.dot(np.transpose(np.mat(x[i]-mu[k])),np.mat((x[i]-mu[k])))
    return(sigma_k/np.sum(p_z_x[:,k]))


# In[28]:

def em(x,K,alpha,mu,sigma,tied):
    #p(z|x)
    p_z_x = np.zeros((N,K),dtype='float64')
    #E-step
    for n in range(N):
        p_x = 0
        for k in range(K):
            p_x += alpha[k]*gaussian(x[n],mu[k],sigma[k])
        for k in range(K):
            p_z_x[n][k] = alpha[k]*gaussian(x[n],mu[k],sigma[k])/p_x
    #M-step
    for k in range(K):
        alpha[k] = new_alpha(p_z_x,k,N)
        mu[k] = new_mu(x,p_z_x,k,N)
        sigma[k] = new_sigma(x,p_z_x,k,N,mu)
    if tied == 1:
        tempt = 0
        for i in range(K):
            tempt += alpha[i]*sigma[i]
        sigma[:] = tempt    
    return(alpha,mu,sigma,p_z_x)


# In[29]:

#likelihood
def likelihood(x,K,alpha,mu,sigma,tied):
    alpha,mu,sigma,p_z_x = em(x,K,alpha,mu,sigma,tied)
    log_likelihood = 0
    for n in range(N):
        p_x = 0
        for k in range(K):
            p_x += alpha[k]*gaussian(x[n],mu[k],sigma[k])
        log_likelihood += math.log(p_x)
    return(log_likelihood,alpha,mu,sigma)


# In[35]:

iteration = 40
K_num = 5


# In[36]:

N = np.shape(x_train)[0]


# In[37]:

def compare(tied):
    likelihoods_train = np.zeros((K_num,iteration))
    likelihoods_dev = np.zeros((K_num,iteration))
    for K in range(3,K_num+3):
        #Initialize parameters
        mu = np.zeros((K,2))
        for k in range(K):
            mu[k]=x[k*50]
        sigma = np.array([[[1, 0], [0, 1]] for i in range(K)], dtype='float64')
        alpha = np.array([1.0/K for i in range(K)], dtype='float64')
        for i in range(iteration):
            #pdb.set_trace()
            likelihoods_train[K-3,i],alpha,mu,sigma = likelihood(x_train,K,alpha,mu,sigma,tied)
            log_likelihood = 0
            for n in range(np.shape(x_dev)[0]):
                p_x = 0
                for k in range(K):
                    p_x += alpha[k]*gaussian(x_dev[n],mu[k],sigma[k])
                likelihoods_dev[K-3,i] += math.log(p_x)
    return(likelihoods_train,likelihoods_dev)


# In[38]:

likelihoods_train_tied,likelihoods_dev_tied = compare(1)
likelihoods_train_separate,likelihoods_dev_separate = compare(0)


# In[17]:

print likelihoods_train_tied


# In[18]:

print likelihoods_dev_tied


# In[19]:

print likelihoods_train_separate


# In[20]:

print likelihoods_dev_separate


# plt.figure(figsize = (15,10))
# plt.subplot(221)
# plt.plot(likelihoods_train_tied[0,:],'r',label='mixture 3')
# plt.plot(likelihoods_train_tied[1,:],'b',label='mixture 4')
# plt.plot(likelihoods_train_tied[2,:],'y',label='mixture 5')
# plt.plot(likelihoods_train_tied[3,:],'g',label='mixture 6')
# plt.plot(likelihoods_train_tied[4,:],'black',label='mixture 7')
# plt.title('train data with tied convariance')
# plt.xlabel('iteration')
# plt.ylabel('log likelihood')
# plt.legend(loc=0)
# plt.subplot(222)
# plt.plot(likelihoods_dev_tied[0,:],'r',label='mixture 3')
# plt.plot(likelihoods_dev_tied[1,:],'b',label='mixture 4')
# plt.plot(likelihoods_dev_tied[2,:],'y',label='mixture 5')
# plt.plot(likelihoods_dev_tied[3,:],'g',label='mixture 6')
# plt.plot(likelihoods_dev_tied[4,:],'black',label='mixture 7')
# plt.title('dev data with tied convariance')
# plt.xlabel('iteration')
# plt.ylabel('log likelihood')
# plt.legend(loc=0)
# plt.subplot(223)
# plt.plot(likelihoods_train_separate[0,:],'r',label='mixture 3')
# plt.plot(likelihoods_train_separate[1,:],'b',label='mixture 4')
# plt.plot(likelihoods_train_separate[2,:],'y',label='mixture 5')
# plt.plot(likelihoods_train_separate[3,:],'g',label='mixture 6')
# plt.plot(likelihoods_train_separate[4,:],'black',label='mixture 7')
# plt.title('train data with separate convariance')
# plt.xlabel('iteration')
# plt.ylabel('log likelihood')
# plt.legend(loc=0)
# plt.subplot(224)
# plt.plot(likelihoods_dev_separate[0,:],'r',label='mixture 3')
# plt.plot(likelihoods_dev_separate[1,:],'b',label='mixture 4')
# plt.plot(likelihoods_dev_separate[2,:],'y',label='mixture 5')
# plt.plot(likelihoods_dev_separate[3,:],'g',label='mixture 6')
# plt.plot(likelihoods_dev_separate[4,:],'black',label='mixture 7')
# plt.title('dev data with separate convariance')
# plt.xlabel('iteration')
# plt.ylabel('log likelihood')
# plt.legend(loc=0)

# In[ ]:



