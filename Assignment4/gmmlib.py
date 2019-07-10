import math 
from scipy.stats import multivariate_normal


def expectation(data, gmmcp):
    N = data.shape[0]
    K = len(gmmcp)
    
    posterior = np.zeros((N,K))
    #cal the multivariate gaussian distributions and likehood x prior
    for k in range(K):
        posterior[:, k] = gmmcp[k]['prior'] * multivariate_normal(mean = gmmcp[k]['mean'], cov = gmmcp[k]['covariance']).pdf(data)
       
    #cal (likehood x prior) / sum(likehood x prior)
    for i in range(N):
        if np.sum(posterior[i,:]) == 0:
            continue
        posterior[i,:] = posterior[i, :]/np.sum(posterior[i,:])
        
    return posterior

def maximizationmean(posterior,data, gmmcp):
    
    N,D = data.shape  
    K = len(gmmcp) 
    mu = np.zeros((K,D))
    cov = []
    prior = np.zeros(K)

    for k in range(K):
        Nk = np.sum(posterior[:,k])
        if Nk == 0:
            mu[k] = gmmcp[k]['mean']
            cov.append(gmmcp[k]['covariance'])
            prior[i] = gmmcp[k]['prior']
            continue

        for d in range(D):
            mu[k,d] = np.sum(posterior[:,k] * data[:,d]) / Nk


        covi = np.mat(np.zeros((D,D)))
                      
        for i in range(N):
            covi += posterior[i,k] * np.mat(data[i] - mu[k]).T * np.mat(data[i] - mu[k])/ Nk

        cov.append(covi)

        prior[k] = Nk/N

    cov = np.array(cov)
    
    gmmcp = [{'mean': mu[k], 'covariance': sigma[k], 'prior': prior[k]} for k in range(K)]
    return gmmcp


def maximization(posterior,data, gmmcp):
    
    N,D = data.shape  
    K = len(gmmcp) 
    mu = np.zeros((K,D))
    cov = []
    prior = np.zeros(K)

    for k in range(K):
        Nk = np.sum(posterior[:,k])
        if Nk == 0:
            mu[k] = gmmcp[k]['mean']
            cov.append(gmmcp[k]['covariance'])
            prior[i] = gmmcp[k]['prior']
            continue

        for d in range(D):
            mu[k,d] = np.sum(posterior[:,k] * data[:,d]) / Nk


        covi = np.mat(np.zeros((D,D)))
                      
        for i in range(N):
            covi += posterior[i,k] * np.mat(data[i] - mu[k]).T * np.mat(data[i] - mu[k])/ Nk

        cov.append(covi)

        prior[k] = Nk/N

    cov = np.array(cov)
    

    gmmcp = [{'mean': mu[k], 'covariance': cov[k], 'prior': prior[k]} for k in range(K)]
    return gmmcp