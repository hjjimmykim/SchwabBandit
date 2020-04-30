import numpy as np
import matplotlib.pyplot as plt

steps = 1000

#p = np.random.random(1)[0]
p=0.65
results = np.random.binomial(n=1, p=p, size=steps)

t = np.arange(steps+1)

mu0 = 0.5
var0 = 1
s0 = 0.5

## Mean/Var Estimates
true_mu = p * np.ones(steps+1)
true_var = p * (1-p) * np.ones(steps+1)

wins = np.cumsum(results)
tot = t[1:]

# Bernoulli
bern_mu = np.append( [mu0], wins/tot )
bern_var = np.append( [var0], ( tot**2 * wins - tot * wins**2 ) / (tot**3) ) + 1E-4

# Gauss 
gauss_mu = mu0 * np.ones(steps+1)
gauss_var = var0 * np.ones(steps+1)
s = s0 * np.ones(steps+1)
for i in range(1,steps+1):
    gauss_mu[i] = ( s[i-1]**2 * gauss_mu[i-1] + gauss_var[i-1] * results[i-1] )/(gauss_var[i-1] + s[i-1]**2 )
    gauss_var[i] = ( gauss_var[i-1] * s[i-1]**2 )/(gauss_var[i-1] + s[i-1]**2 )

'''
# Gauss Smooth
n = 10
gauss2_mu = mu0 * np.ones(steps+1)
gauss2_var = var0 * np.ones(steps+1)
for i in range(1,int(steps/n)):
    gauss2_mu[n*i:n*(i+1)] = ( s[n*i-1]**2 * gauss2_mu[n*i-1] + gauss2_var[n*i-1] * np.sum(results[n*i-1:n*(i+1)-1]) )/(n * gauss2_var[n*i-1] + s[n*i-1]**2 )
    gauss2_var[n*i:n*(i+1)] = ( gauss2_var[n*i-1] * s[n*i-1]**2 )/(n * gauss2_var[n*i-1] + s[n*i-1]**2 )
'''

## Plot means
plt.figure()

plt.plot( t, bern_mu, 'r', label="Bernoulli" )
plt.plot( t, gauss_mu, 'b', label="Gauss" )
#plt.plot( t, gauss2_mu, 'm', label="Gauss 2" )
plt.plot( t, true_mu, 'g', label="True" )

plt.legend(loc='center right')
plt.xlabel("Steps")
plt.ylabel("Mean")
plt.title(str(steps) + " Bernoulli trials with parameter " + str(round(p,4)) )

plt.show()


'''
## Plot vars
plt.figure()

plt.plot( t, s, 'r', label="Bernoulli" )
plt.plot( t, gauss_var, 'b', label="Gauss" )
plt.plot( t, true_var, 'g', label="True" )

plt.legend(loc='center right')
plt.xlabel("Steps")
plt.ylabel("Var")
plt.title(str(steps) + " Bernoulli trials with parameter " + str(round(p,4)) )

plt.show()
'''
