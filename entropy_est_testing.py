import timeit
import numpy as np
import matplotlib.pyplot as plt

import entropy_estimators as ee
import bandit as Ban #Insert 7 Deadly Sins anime reference


timing_runs = 10000   # Number of runs to determine computation time
acc_runs = 10000      # Number of runs to determine accuracy


## Testing Sampling
N_disc = 50 # Number of discrete points
N_sample = 100000 # Number of samples
x = np.arange(0,1,0.01)
plt.plot(x, Ban.Rhomax(x,2,5,5,2), color='r')

# Sampling
Rhomax_vec = np.vectorize(lambda p: Ban.Rhomax(p, 2, 5, 5, 2))
p_list = np.linspace(0,1,N_disc+2)[1:-1]
# Discretize
Rhomax_disc = Rhomax_vec(p_list)
# Normalize
Rhomax_disc = Rhomax_disc/np.sum(Rhomax_disc)
# Sample from the distribution
sample_points = np.random.choice(p_list, N_sample, p=Rhomax_disc)

n, bins, patches = plt.hist(x=sample_points, bins=p_list, density=True)
maxfreq = n.max()
plt.show()


##Timing
'''
int_time = timeit.timeit(stmt = 'Ban.Entropy_int(2,5,5,2)',
        setup = 'import bandit as Ban', number=timing_runs )

print("Full Integral:", int_time)


est_time = timeit.timeit(stmt = 'Ban.Entropy_est(2,5,5,2)',
        setup = 'import bandit as Ban', number=timing_runs )

print("Jimmy's Estimator:", est_time)
'''
'''
## Accuracy
ent_target = Ban.Entropy_int(2,5,5,2)
J_est_list = np.array([])
for i in range(acc_runs):
    J_est_list = np.append( J_est_list, [Ban.Entropy_est(2,5,5,2)] )
J_est_mean = np.mean(J_est_list)
#J_est_std = np.std(J_est_list)
J_MSE = np.mean( (J_est_list-ent_target)**2 )
print("Jimmy's Mean:", J_est_mean)
print("Jimmy's MSE:", J_MSE)


## Plotting (but not scheming)
n, bins, patches = plt.hist(x=J_est_list, bins='auto')
#plt.axvline( x=ent_target, color='r', linewidth=1, linestyle='--' )
plt.axvline( x=J_est_mean, color='b', linewidth=1, linestyle='--' )
plt.xlabel('Entropy')
plt.ylabel('Frequency')
plt.title("Jimmy's Estimated Entropy")
maxfreq = n.max()
plt.show()
'''
