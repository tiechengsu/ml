Name: Tiecheng Su
Email: tsu5@ur.rochester.edu
Course: ECE 446
Implement EM to train an HMM for whichever dataset you used for assignment 7. The observation probs should be as in assignment 7: either gaussian, 
or two discrete distributions conditionally independent given the hidden state. Does the HMM model the data better than the original non-sequence model? 
What is the best number of states?
************ Files *********
em_hmm.py
README
compare.png
************ Algorithm *****
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

************ Instructions ***
python HW8.py

************ Results ********
the log_likelihood of mixture 3,4,5,6
[-2327.17359576 -2188.55451226 -2335.55487979 -2183.92273655]
According to the results, the log liklihood is much better than the original non-sequence model. When the states number equals 4 and 6, we can get the best result.

************ Your interpretation *******
The sequence model can better describe the relationship between the data points. As for the original non-sequence model,it suppose each datapoints are i.i.d 

************ References ************
People that I discussed with
Yanfu Zhang
