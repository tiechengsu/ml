Name: Tiecheng Su
Email: tsu5@ur.rochester.edu
Course: ECE 446
Implement EM fitting of a mixture of gaussians on the two-dimensional data set points.dat. You should try different numbers of mixtures, as well as tied vs. separate covariance matrices for each gaussian. OR Implement EM fitting of the aspect model on the discrete data set pairs.dat. You should try different numbers of mixtures.

IN EITHER CASE Use the final 1/10 of the data for dev. Plot likelihood on train and dev vs iteration for different numbers of mixtures.

************ Files *********
em_gmm.py
README
compare.png

************ Algorithm *****
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

def gaussian(x,mu,sigma):
    g = np.exp(-0.5*np.dot(np.dot(np.transpose(x-mu),np.linalg.pinv(sigma)),(x-mu)))/(2*math.pi*math.sqrt(np.linalg.det(sigma)))
    return(g)

def new_alpha(p_z_x,k,N):
    return(np.sum(p_z_x[:,k])/N)

def new_mu(x,p_z_x,k,N):
    mu_k = np.zeros((1,2),dtype='float64')
    for i in range(N):
        mu_k += p_z_x[i][k]*x[i]
    return mu_k/np.sum(p_z_x[:,k])

def new_sigma(x,p_z_x,k,N,mu):
    sigma_k = np.zeros((2,2),dtype='float64')
    for i in range(N):
        sigma_k += p_z_x[i][k]*np.dot(np.transpose(np.mat(x[i]-mu[k])),np.mat((x[i]-mu[k])))
    return(sigma_k/np.sum(p_z_x[:,k]))

************ Instructions ***
python hw7.py

************ Results ********
From the compare result we find out that for the train dataset, it will converge eventually. For mixture4-7, it will converge to the same log likelihood
value, but higher mixture number will lead to faster converge speed. And the mixture will converge to a lower log likelihood. For the data with separate
convariance matrix, it will converge slower than the tied one. For the dev dataset, it will reach its best log likelihood, and then it will decrease.

************ Your interpretation *******
The separate convariance is used to prevent overfit. As for dev dataset doesn't converge, it's because it use the variable calculated from the train dataset.

************ References ************
People that I discussed with
Yanfu Zhang
