import time
import timeit
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import beta
from scipy.integrate import quad

from numba import complex128, float64, jit

import entropy_estimators as ee
import bandit as Ban #Insert 7 Deadly Sins anime reference


timing_runs = 1000   # Number of runs to determine computation time
acc_runs = 10000      # Number of runs to determine accuracy


## Testing Sampling
'''
N_disc = 50 # Number of discrete points
N_sample = 10000 # Number of samples
x = np.arange(0,1,0.01)
plt.plot(x, Ban.Rhomax(x,2,5,5,2), color='r')

Rhomax_vec = np.vectorize(lambda p: Ban.Rhomax(p, 2, 5, 5, 2))
p_list = np.linspace(0,1,N_disc+2)[1:-1]
# Discretize
Rhomax_disc = Rhomax_vec(p_list)
# Normalize
Rhomax_disc = Rhomax_disc/np.sum(Rhomax_disc)
# Sample from the distribution
sample_lst = []
for i in range(N_sample):
    sample_lst.append( np.random.choice(p_list, p=Rhomax_disc) )
#sample_lst = np.random.choice(p_list, N_sample, p=Rhomax_disc)

plt.hist(x=sample_lst, bins=p_list, density=True)
#maxfreq = n.max()
plt.show()
'''

'''
# Jimmy Estimation
J_est_list = np.array([])
for i in range(acc_runs):
    J_est_list = np.append( J_est_list, [Ban.Entropy_est(2,5,5,2)] )
J_est_mean = np.mean(J_est_list)
#J_est_std = np.std(J_est_list)
J_MSE = np.mean( (J_est_list-ent_target)**2 )
print("Jimmy's Mean:", J_est_mean)
print("Jimmy's MSE:", J_MSE)
'''
# Nearest Neighbor Entropy Estimation
def nn_entropy(a1,a2,b1,b2, N_disc=50, N_sample=10000):
    Rhomax_vec = np.vectorize(lambda p: Ban.Rhomax(p, a1, a2, b1, b2))
    p_list = np.linspace(0,1,N_disc+2)[1:-1]
    # Discretize
    Rhomax_disc = Rhomax_vec(p_list)
    # Normalize
    Rhomax_disc_nrm = Rhomax_disc/np.sum(Rhomax_disc)
    # Sample from the distribution
    sample_lst = []
    for i in range(N_sample):
        sample_lst.append( [np.random.choice(Rhomax_disc, p=Rhomax_disc_nrm)] )
    return ee.entropy(sample_lst)

# Modified Jimmy Estimation
def entropy_est2(a1,a2,b1,b2, N_disc=50, N_sample=10000):
    Rhomax_vec = np.vectorize(lambda p: Ban.Rhomax(p, a1, a2, b1, b2))
    p_list = np.linspace(0,1,N_disc+2)[1:-1]
    # Discretize
    Rhomax_disc = Rhomax_vec(p_list)
    # Normalize
    Rhomax_disc_nrm = Rhomax_disc/np.sum(Rhomax_disc)
    # Sample from the distribution
    sample_lst = np.array([])
    sample_lst = np.append( sample_lst, np.random.choice(-np.log2(Rhomax_disc), p=Rhomax_disc_nrm) )
    return np.mean(sample_lst)

# Playing around with numeric integration
#@jit(complex128(float64, float64, float64,float64,float64), nopython=True, cache=True)
def rhomax_integrand(x,a1,a2,b1,b2):
    pdf1 = beta.pdf(x,a1,b1)
    pdf2 = beta.pdf(x,a2,b2)
    cdf1 = beta.cdf(x,a1,b1)
    cdf2 = beta.cdf(x,a2,b2)
    
    rho = pdf1 * cdf2 + pdf2 * cdf1

    integrand = -rho * np.log( rho )
    return integrand
def entropy_int2(integrand, arg):
    integral = quad(integrand, 0, 1, args=(arg))[0]
    return integral

def entropy_int3(a1,a2,b1,b2):
    def integrand(x):
        pdf1 = beta.pdf(x,a1,b1)
        pdf2 = beta.pdf(x,a2,b2)
        cdf1 = beta.cdf(x,a1,b1)
        cdf2 = beta.cdf(x,a2,b2)
        rho = pdf1 * cdf2 + pdf2 * cdf1
        return -rho * np.log(rho)
    integral = quad(integrand, 0, 1)[0]
    return integral


##Timing
def time_test(fnc, args, trials=timing_runs):
    t_start = time.time()
    for i in range(trials):
        fnc(*args)
    t_fin = time.time()
    return t_fin-t_start
print("Integral 1:", time_test(Ban.Entropy_int, (2,5,5,2) ) )
print("Integral 2:", time_test(entropy_int2, (rhomax_integrand,(2,5,5,2)) ) )
print("Integral 3:", time_test(entropy_int3, (2,5,5,2) ) )

## Accuracy
ent_target = Ban.Entropy_int(2,5,5,2)
def acc_test(fnc, args, trials=acc_runs):
    ent_est_lst = np.array([])
    for i in range(trials):
        ent_est_lst = np.append( ent_est_lst, fnc(*args) )
    ent_est_mean = np.mean(ent_est_lst)
    MSE = np.mean( (ent_est_lst-ent_target)**2 )
    return ent_est_mean, MSE
print("Integral 1:", acc_test(Ban.Entropy_int, (2,5,5,2) ) )
print("Integral 2:", acc_test(entropy_int2, (rhomax_integrand,(2,5,5,2)) ) )
print("Integral 3:", acc_test(entropy_int3, (2,5,5,2) ) )
'''
## Plotting (but not scheming)
plt.hist(x=J_est_list, bins='auto')
#plt.axvline( x=ent_target, color='r', linewidth=1, linestyle='--' )
plt.axvline( x=J_est_mean, color='b', linewidth=1, linestyle='--' )
plt.xlabel('Entropy')
plt.ylabel('Frequency')
plt.title("Jimmy's Estimated Entropy")
plt.show()

plt.hist(x=ent_est_lst, bins='auto')
plt.axvline( x=ent_target, color='r', linewidth=1, linestyle='--' )
plt.axvline( x=ent_est_mean, color='b', linewidth=1, linestyle='--' )
plt.xlabel('Entropy')
plt.ylabel('Frequency')
plt.title("Estimated Entropy")
plt.show()
'''
