import timeit
import numpy as np
import matplotlib.pyplot as plt

import entropy_estimators as ee
import bandit as Ban #Insert 7 Deadly Sins anime reference


timing_runs = 10000   # Number of runs to determine computation time
acc_runs = 1000      # Number of runs to determine accuracy


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

##Timing
'''
int_time = timeit.timeit(stmt = 'Ban.Entropy_int(2,5,5,2)',
        setup = 'import bandit as Ban', number=timing_runs )

print("Full Integral:", int_time)


est_time = timeit.timeit(stmt = 'Ban.Entropy_est(2,5,5,2)',
        setup = 'import bandit as Ban', number=timing_runs )

print("Jimmy's Estimator:", est_time)
'''

## Accuracy
ent_target = Ban.Entropy_int(2,5,5,2)
print("Target Entropy:", ent_target)

# Jimmy Estimation
'''
J_est_list = np.array([])
for i in range(acc_runs):
    J_est_list = np.append( J_est_list, [Ban.Entropy_est(2,5,5,2)] )
J_est_mean = np.mean(J_est_list)
#J_est_std = np.std(J_est_list)
J_MSE = np.mean( (J_est_list-ent_target)**2 )
print("Jimmy's Mean:", J_est_mean)
print("Jimmy's MSE:", J_MSE)
'''
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
##        sample_lst.append( np.random.choice(Rhomax_disc, p=Rhomax_disc_nrm) )
    return ee.entropy(sample_lst)
'''
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

ent_est_lst = np.array([])
for i in range(acc_runs):
    ent_est_lst = np.append( ent_est_lst, entropy_est2(2,5,5,2) )
ent_est_mean = np.mean(ent_est_lst)
MSE = np.mean( (ent_est_lst-ent_target)**2 )
print("Mean:", ent_est_mean)
print("MSE:", MSE)


## Plotting (but not scheming)
'''
plt.hist(x=J_est_list, bins='auto')
#plt.axvline( x=ent_target, color='r', linewidth=1, linestyle='--' )
plt.axvline( x=J_est_mean, color='b', linewidth=1, linestyle='--' )
plt.xlabel('Entropy')
plt.ylabel('Frequency')
plt.title("Jimmy's Estimated Entropy")
plt.show()
'''
plt.hist(x=ent_est_lst, bins='auto')
plt.axvline( x=ent_target, color='r', linewidth=1, linestyle='--' )
plt.axvline( x=ent_est_mean, color='b', linewidth=1, linestyle='--' )
plt.xlabel('Entropy')
plt.ylabel('Frequency')
plt.title("Estimated Entropy")
plt.show()
