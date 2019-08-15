import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.stats import beta

from scipy.integrate import quad
import ghalton

import time
import datetime
import argparse
import pickle

# Information Theory Functions ------------------------------------------------------------------

# KL-divergence
def KLdivergence(p,q):
    # Takes two probability numbers and returns a number
    return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

# Probability distribution over max probability
def rhomax_integrand(x, a1, a2, b1, b2):
    # Takes numbers and returns a prob. number
    
    pdf1 = beta.pdf(x,a1,b1)
    pdf2 = beta.pdf(x,a2,b2)
    cdf1 = beta.cdf(x,a1,b1)
    cdf2 = beta.cdf(x,a2,b2)
    
    rho = pdf1 * cdf2 + pdf2 * cdf1
    
    integrand = -rho * np.log( rho )
    return integrand


# Posterior distribution over outcome given posterior over probs.
def Prob(x, a, b):
    # Takes x = 0 or 1 and a,b = numbers; returns a prob. number
    if x == 1: # Success
        return a/(a+b)
    elif x == 0:
        return b/(a+b)
        
# Estimate entropy via sampling
seq = ghalton.Halton(1)
def Entropy_est(a1, a2, b1, b2, samples=100):
    seq.reset()
    x = seq.get(samples)
    x = np.reshape(x, samples)
    integral = np.mean( rhomax_integrand(x,a1,a2,b1,b2) )
    return integral


# Main function ---------------------------------------------------------------------------------

def main(args):
    # Input unpacking
    p1 = args.p1            # Success probability of the first arm
    p2 = args.p2            # Success probability of the second arm
    n = args.n              # Total number of plays
    N = args.N              # Batch/replicas (for averaging)
    n_rec = args.n_rec      # Record every n plays
    algo = args.algo        # 'Thompson', 'Infomax'
    seed = args.seed        # Random seed
    verbose = args.verbose  # Print messages (time)
    
    # Set random seed
    np.random.seed(seed)
    
    # Input representation
    p = np.array([p1,p2])             # Prob. of arms
    
    # Output initialization
    curr_cum_reward = np.zeros(N)
    curr_cum_subopt = np.zeros(N) # Number of times you pull suboptimal arm
    cum_reward = np.zeros([N,int(n/n_rec)]) # Record cumulative reward at each step
    cum_subopt = np.zeros([N,int(n/n_rec)]) # Record number of suboptimal plays
    plays = np.zeros(int(n/n_rec))
    
    # Setup
    if algo == 'Thompson':
        # Initialize beta distribution parameters
        # beta_params_(batch, {alpha,beta}, arm)
        # 0.5 = uninformative prior, 1 = uniform prior
        beta_params = 1*np.ones([N,2,2]) # [[[a1,a2],[b1,b2]], ...]
    elif algo in ['Infomax', 'Infomax_est']:
        beta_params = 1*np.ones([N,2,2])
        if algo == 'Infomax':
            Entropy = Entropy_int
        elif algo == 'Infomax_est':
            Entropy = Entropy_est
    
    # Simulation
    t_start = time.time()
    t1 = time.time()
    for t in range(n):
    
        # Choose arm
        if algo == 'Thompson':
            theta = np.random.beta(beta_params[:,0,:],beta_params[:,1,:]) # [batch, arm]
            action = np.argmax(theta,1) # Choose the arm corresponding to the optimal draw (N-array)
        elif algo in ['Infomax', 'Infomax_est']:
            action = np.zeros(N, dtype = np.int8)
            for i in range(N): # Loop through batch
                a1, b1 = beta_params[i,:,0] # Arm 0
                a2, b2 = beta_params[i,:,1] # Arm 1
                
                # Original entropy (don't need for comparing difference)
                #Entropy0 = Entropy(a1, a2, b1, b2)
                
                # Arm 0
                # Difference in entropy when 0 (failure) observed
                delH0_0 = Entropy(a1, a2, b1 + 1, b2) #- Entropy0
                # Difference in entropy when 1 (success) observed
                delH0_1 = Entropy(a1 + 1, a2, b1, b2) #- Entropy0
                # Expected decrease in entropy
                dH0 = Prob(0,a1,b1)*delH0_0 + Prob(1,a1,b1)*delH0_1
                
                # Arm 1
                # Difference in entropy when 0 (failure) observed
                delH1_0 = Entropy(a1, a2, b1, b2 + 1) #- Entropy0
                # Difference in entropy when 1 (success) observed
                delH1_1 = Entropy(a1, a2 + 1, b1, b2) #- Entropy0
                # Expected decrease in entropy
                dH1 = Prob(0,a2,b2)*delH1_0 + Prob(1,a2,b2)*delH1_1
                
                action[i] = int(dH0 > dH1) # pick action that decreases H more (i.e. dH more negative)
            
        # Record suboptimal plays
        curr_cum_subopt += action
        if t % n_rec == 0:
            cum_subopt[:,int(t/n_rec)] = curr_cum_subopt
        #if t > 0:
        #    cum_subopt[:,t] = cum_subopt[:,t-1]
        #cum_subopt[:,t] += action

        # Play the arm
        chance = np.random.random(N) # Random number between 0 and 1
        r = (p[action] > chance).astype(int)         # Get reward
        
        # Keep track of cumulative rewards
        curr_cum_reward += r
        # Record cumulative rewards
        if t % n_rec == 0:
            cum_reward[:,int(t/n_rec)] = curr_cum_reward
            plays[int(t/n_rec)] = t+1
            
        # Update parameters
        if algo in ['Thompson', 'Infomax', 'Infomax_est']: # Same posterior updates for both
            # Update a
            beta_params[:,0,:][np.arange(N),action] += r
            # Update b
            beta_params[:,1,:][np.arange(N),action] += 1-r
        
        # Time
        if verbose:
            if (t+1) % int(n/10) == 0:
                t2 = time.time()
                print('Runtime for plays ' + str(t-(int(n/10)-1)) + ' - ' + str(t) + ': ' + str(t2-t1) + ' s')
                t1 = t2
                
    t_end = time.time()
        
    output = {'cum_reward':cum_reward, 'cum_subopt':cum_subopt, 'beta_params':beta_params, 'plays': plays}

    if verbose:
        print('Total runtime: ' + str(t_end-t_start) + ' s')
        print("Cumulative Reward:", cum_reward[0][-1])
        
    return output
    
# Plotting functions ----------------------------------------------------------------------------
    
# Plot learned beta distribution
def plot_beta(x, a1, a2, b1, b2, p1, p2, n, name, savefig):
    # ai, bi's = beta distribution parameters for each arm
    
    plt.figure()
    beta1 = beta.pdf(x, a1, b1)
    beta2 = beta.pdf(x, a2, b2)
    plt.plot(x, beta1, 'r', linewidth=3, label='Superior arm')
    plt.plot(x, beta2, 'b', linewidth=3, label='Inferior arm')
    plt.axvline(x=p1,color='r', linewidth=1, linestyle='--')
    plt.axvline(x=p2,color='b', linewidth=1, linestyle='--')
    plt.legend(loc='best')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$p(\theta)$')
    plt.title('Learned beta distributions for each arm after ' + str(n) + ' plays')
    if savefig:
        plt.savefig('beta_distribution_' + name + '.png')
        plt.close()
    else:
        plt.show()
        
# Plot regret
def plot_regret(plays, c_mean, p1, p2, N, name, savefig, savedata, algo):
    plt.figure()
    # Optimal expected number of suboptimal plays (Lai-Robbins lower bound)
    LB = np.log(plays)/KLdivergence(p1,p2)
    # Optimal regret
    regret_opt = np.maximum(LB * (p1 - p2)-20, np.zeros(LB.shape))
    # Actual regret
    regret_act = c_mean * (p1 - p2)
    plt.plot(plays, regret_opt, 'k', linewidth=3, label='Lai-Robbins bound slope')
    plt.plot(plays, regret_act, 'r', linewidth=3, label=algo)
    plt.legend(loc='best')
    plt.gca().set_xscale('log',basex=10)
    plt.ylim([0, np.ceil(np.amax(regret_act)) ]) # Adaptive max of graph
    plt.xlabel('Total number of plays')
    plt.ylabel(r'Regret $<n_2>(p_1-p_2)$')
    plt.title('Regret averaged over ' + str(N) + ' replicas')
    if savedata:
        savePath = 'Save/'
        save_time = datetime.datetime.today().strftime('%Y-%m-%d_%H:%M')
        pickle_name = save_time + '.pkl'
        with open(savePath + pickle_name, 'wb') as f:
            pickle.dump(regret_act, f)
    if savefig:
        plt.savefig('regret_' + name + '.png')
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # Parse inputs ------------------------------------------------------------------------------
    # Example command: python bandit.py --n 100 --algo Infomax --verbose 1
    parser = argparse.ArgumentParser(description='2-armed bandit')
    
    parser.add_argument("--p1", default=0.9, type=float, help="Success probability of the superior arm")
    parser.add_argument("--p2", default=0.8, type=float, help="Success probability of the inferior arm")
    parser.add_argument("--n", default=1000, type=int, help="Total number of plays")
    parser.add_argument("--N", default=1, type=int, help="Replicas (For averaging)")
    parser.add_argument("--n_rec", default=1, type=int, help="Record every n plays")
    parser.add_argument("--algo", default='Thompson', type=str, help="Decision algorithm; 'Thompson', 'Infomax'")
    parser.add_argument("--seed", default=111, type=int, help="Random seed")
    parser.add_argument("--savefig", default=0, type=int, help="Save figures")
    parser.add_argument("--savedata", default=0, type=int, help="Save regret data")
    parser.add_argument("--verbose", default=0, type=int, help="Print messages")
    
    args = parser.parse_args()

    # Run main function -------------------------------------------------------------------------
    output = main(args)
    
    # Save output -------------------------------------------------------------------------------

    name =  'p1=' + str(args.p1) + \
            '_p2=' + str(args.p2) + \
            '_n=' + str(args.n) + \
            '_N=' + str(args.N) + \
            '_n_rec=' + str(args.n_rec) + \
            '_algo=' + str(args.algo) + \
            '_seed=' + str(args.seed)
                
    #pickle.dump(output, open(name + '.pickle',"wb"))
    
    # Extract outputs ---------------------------------------------------------------------------
    a = output['cum_reward']
    b = output['beta_params']
    c = output['cum_subopt']
    plays = output['plays']
    c_mean = np.mean(c,0) # Take average over batches

    # Plot beta distribution --------------------------------------------------------------------
    x = np.linspace(0,1,1000) # Parameter space [0,1]
    plot_beta(x,b[0,0,0],b[0,0,1],b[0,1,0],b[0,1,1],args.p1,args.p2,args.n,name,args.savefig)

    # Plot regret -------------------------------------------------------------------------------
    plot_regret(plays,c_mean,args.p1,args.p2,args.N,name,args.savefig, args.savedata, args.algo)
