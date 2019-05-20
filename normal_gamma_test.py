# Model a Gaussian with unknown means & std with a normal-gamma distribution

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.stats import gamma 
from scipy.stats import t as tdist
from scipy.special import gamma as gammafunc

# Marginal distribution for mu
def pmu(mu, params):
    mu0 = params[0]
    lamb = params[1]
    alpha = params[2]
    beta = params[3]

    return (lamb/(2*np.pi))**0.5 * beta**alpha * gammafunc(alpha+0.5)/gammafunc(alpha) * (beta + 0.5*lamb*(mu-mu0)**2)**(-(alpha+0.5))
    
# Gamma distribution for tau
def ptau(tau, params):
    alpha = params[2]
    beta = params[3]

    return gamma.pdf(tau, alpha, scale=1/beta)

mu_true = 5
tau_true = 10 # Precision (inverse variance)

sig_true = 1/np.sqrt(tau_true) # Std

N = 1 # Number of episodes (update iteration)
n = 100   # Number of samples to collect each episode

M1 = 0 # Incremental average
M2 = 0 # Incremental 2nd moment average

# Parameters (mu0, lambda, alpha, beta)
NG_params = 1.5*np.ones([1,4,1])

# Simulation
for i in range(N):
    for j in range(n):
        # Sample
        x = np.random.normal(mu_true,sig_true)
        
        # Update incremental averages
        M1 = M1 + (x - M1)/(j+1)
        M2 = M2 + (x**2 - M2)/(j+1)
        
    M1_true = NG_params[0][0]
    M2_true = (NG_params[0][1]+1)/NG_params[0][1] * NG_params[0][3]/(NG_params[0][2]-1) + NG_params[0][0]**2
    
    # Check if Lemma 3.4 in Bayesian Q-learning paper is roughly correct
    print('M1 = ' + str(M1))
    print('M1_true = ' + str(M1_true))
    print('M2 = ' + str(M2))
    print('M2_true = ' + str(M2_true))
    
    # Update parameters
    mu0,lamb,alpha,beta = NG_params[0]
    
    NG_params[0][0] = (lamb*mu0 + n*M1)/(lamb + n)
    NG_params[0][1] += n
    NG_params[0][2] += 0.5*n
    NG_params[0][3] += 0.5*n*(M2-M1**2) + (n*lamb*(M1-mu0)**2)/(2*(lamb+n))
   
# Sample from normal-gamma distribution
N_sample = 100 # Number of samples to collect

# Sample tau (shape, rate)
tau_sample = np.random.gamma(NG_params[0][2], 1/NG_params[0][3], N_sample)

# Sample mu (mean, std)
mu_sample = np.random.normal(NG_params[0][0],np.sqrt(1/(NG_params[0][1]*tau_sample)))

# Sample mu (Marginal distribution)
mu_sample2 = tdist.rvs(2*NG_params[0][2], NG_params[0][0], np.sqrt(NG_params[0][3]/(NG_params[0][1]*NG_params[0][2])), size=N_sample)

# ----------------------------------------------------------------------- 
#NG_params = [[1.5,1.5,1.5,1.5]]  
# Plot posterior distributions
xmu = np.linspace(-10,10,1000)
xtau = np.linspace(0,20,1000)

plt.figure()
plt.subplot(121)
plt.plot(xmu, pmu(xmu,NG_params[0]))
plt.axvline(x=mu_true,color='r', linewidth=1, linestyle='--')
plt.title(r'Posterior distribution for $\mu$')
plt.xlabel(r'$\mu$')
plt.ylabel('pdf')

plt.subplot(122)
plt.plot(xtau, ptau(xtau,NG_params[0]))
plt.axvline(x=tau_true,color='r', linewidth=1, linestyle='--')
plt.title(r'Posterior distribution for $\tau$')
plt.xlabel(r'$\tau$')
plt.ylabel('pdf')

plt.show()

# Plot sampled mu & tau
plt.figure()
plt.subplot(121)
plt.hist(mu_sample, label=r'Both $\tau$ and $\mu$ sampled')
plt.hist(mu_sample2, label=r'Marginal samples')
plt.axvline(x=mu_true,color='r', linewidth=1, linestyle='--')
plt.title(r'Sampled distribution for $\mu$')
plt.xlabel(r'$\mu$')
plt.ylabel('pdf')
plt.xlim([4,6])
plt.legend()

plt.subplot(122)
plt.hist(tau_sample)
plt.axvline(x=tau_true,color='r', linewidth=1, linestyle='--')
plt.title(r'Sampled distribution for $\tau$')
plt.xlabel(r'$\tau$')
plt.ylabel('pdf')
plt.xlim([0,20])

plt.show()
