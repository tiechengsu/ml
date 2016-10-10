
# coding: utf-8

# In[100]:

import numpy as np
import numpy.matlib
import math
#import matplotlib.pylab as plt
#%pylab inline


# In[63]:

x1 = []
x2 = []
#with open('E:/2016Spring/ML/hw7/points.dat.txt') as f:
with open('points.dat') as f:
    for line in f:
        entries = line.strip().split(' ')
        x1.append(entries[0])
        x2.append(entries[-1])
x = [x1,x2]
x = np.transpose(np.asarray(x,dtype='float64'))
x = x[:np.shape(x)[0]*0.9]


# In[64]:

def gaussian(x,mu,sigma):
    g = np.exp(-0.5*np.dot(np.dot(np.transpose(x-mu),np.linalg.pinv(sigma)),(x-mu)))/float(2*math.pi*math.sqrt(np.linalg.det(sigma)))
    return(g)


# In[65]:

def alpha_beta(A,initialprob,mu,sigma):
    #Forward algorithm
    #alpha = P(x1,...xt,zt=j)
    c = np.zeros((T,1),dtype='float64')#scaling factor
    alpha = np.zeros((S,T),dtype='float64')
    for s in range(S):
        alpha[s,0] = initialprob[s]*gaussian(x[0],mu[s],sigma[s])
        c[0] = np.sum(alpha[:,0])
    alpha[:,0] = alpha[:,0]/float(c[0])
    for t in range(1,T):
        for s in range(S):
            alpha[s,t] = gaussian(x[t],mu[s],sigma[s])*sum([alpha[s2,t-1]*A[s,s2] for s2 in range(S)])
        c[t] = sum([alpha[s,t] for s in range(S)])
        alpha[:,t] = alpha[:,t]/float(c[t])

    #Backward algorithm
    #beta = P(x_{t+1},...,x_T|zt=j)
    beta = np.zeros((S,T),dtype='float64')
    beta[:,T-1] = 1.0
    for t in range(T-2,-1,-1):
        for s in range(S):
            beta[s,t] = sum([beta[s2,t+1]*A[s2,s]*gaussian(x[t+1],mu[s2],sigma[s2]) for s2 in range(S)])
        beta[:,t] = beta[:,t]/float(c[t+1])
    return(c,alpha,beta)


# In[66]:

def new_mu(x,gamma,s,T):
    mu_s = np.zeros((1,2),dtype='float64')
    for t in range(T):
        mu_s += gamma[s][t]*x[t]
    return mu_s/float(np.sum(gamma[s,:]))


# In[67]:

def new_sigma(x,gamma,s,T,mu):
    sigma_s = np.zeros((2,2),dtype='float64')
    for t in range(T):
        sigma_s += gamma[s][t]*np.dot(np.transpose(np.mat(x[t]-mu[s])),np.mat((x[t]-mu[s])))
    return(sigma_s/float(np.sum(gamma[s,:])))


# In[87]:

def em(S,T,A,initialprob,mu,sigma):
    c,alpha,beta = alpha_beta(A,initialprob,mu,sigma)
    #E-Step
    #xi is a list of T matrices of size (S,S)
    #gamma is a (S,T) matrices
    gamma = np.zeros((S,T),dtype='float64')
    xi = []
    for t in range(T):
        xi_t = np.zeros((S,S),dtype='float64')
        for i in range(S):
            gamma[i,t] = alpha[i,t]*beta[i,t]
            for j in range(S):
                xi_t[i,j] = (alpha[j,t-1]*A[j,i]*gaussian(x[t],mu[i],sigma[i])*beta[i,t])
        gamma[:,t] = gamma[:,t]/float(sum([alpha[i,t]*beta[i,t] for i in range(S)]))
        xi_t = xi_t/float(sum(sum(xi_t)))
        xi.append(xi_t)
        ect = np.sum(xi,axis=0)
    #M-Step
    #initial probabilities
    initialprob = np.array([gamma[s_i,0] for s_i in range(S)],dtype='float64')
    #transition probabilities
    for s_i in range(S):
        for s_j in range(S):
            A[s_i,s_j] = ect[s_i,s_j]/float(sum([ect[s_i,s_j] for s_i in range(S)]))
        #emission probabilities
        mu[s_i] = new_mu(x,gamma,s_i,T)
        sigma[s_i] = new_sigma(x,gamma,s_i,T,mu)
    return(c,A,initialprob,mu,sigma)


# In[95]:

def initial(S,cluster):
    #initial probabilities
    initialprob = np.ones((S,1),dtype='float64')*1/float(S)
    #transition probability
    A = np.ones((S,S),dtype='float64')*1/float(S)
    #emission probability
    mu = np.zeros((S,2),dtype='float64')
    sigma = np.zeros((S,2,2),dtype='float64')
    for s in range(S):
        mu[s] = np.mean(x[np.where(cluster == s+1)],axis=0)
        sigma[s] = np.dot(np.transpose(np.mat(x[np.where(cluster == s+1)]-mu[s])),np.mat((x[np.where(cluster == s+1)]-mu[s])))/float(np.shape(x[np.where(cluster == s+1)])[0])
    return(A,initialprob,mu,sigma)


# In[107]:

T = len(x)
iteration = 50
log_likelihood = np.zeros((4,iteration),dtype='float64')
for S in range(3,7):#number of clusters
    cluster = np.random.choice(np.arange(1,S+1),np.shape(x)[0],np.ndarray.tolist(np.ones((1,S),dtype='float64')*1/float(S)))
    A,initialprob,mu,sigma = initial(S,cluster)
    for i in range(iteration):
        c,A,initialprob,mu,sigma = em(S,T,A,initialprob,mu,sigma)
        log_likelihood[S-3,i] = np.sum(np.log(c))


# In[108]:

#print('the log_likelihood of mixture 3,4,5,6')
#print(log_likelihood[:,-1])
print 'the log_likelihood of mixture 3,4,5,6'
print log_likelihood[:,-1]


# plt.figure(figsize = (15,10))
# plt.plot(log_likelihood[0,:],'r',label='mixture 3')
# plt.plot(log_likelihood[1,:],'b',label='mixture 4')
# plt.plot(log_likelihood[2,:],'y',label='mixture 5')
# plt.plot(log_likelihood[3,:],'g',label='mixture 6')
# plt.xlabel('iteration')
# plt.ylabel('log likelihood')
# plt.legend(loc=0)

# In[ ]:



