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

# Posterior distribution over outcome given posterior over probs.
def Beta_Post(outcome, a, b):
    # Takes x = 0 or 1 and a,b = numbers; returns a prob. number
    if outcome:
        return a/(a+b)
    else:
        return b/(a+b)

# Estimate rhomax entropy via sampling
seq = ghalton.Halton(1)
def Beta_Entropy(a0, b0, a1, b1, samples=1000):
    seq.reset()
    x = seq.get(samples)
    x = np.reshape(x, samples)

    pdf0 = beta.pdf(x,a0,b0)
    pdf1 = beta.pdf(x,a1,b1)
    cdf0 = beta.cdf(x,a0,b0)
    cdf1 = beta.cdf(x,a1,b1)
    
    rho = pdf0 * cdf1 + pdf1 * cdf0
    integral = np.mean( -rho * np.log( rho ) )
    return integral


def Beta_Action(params, N):
    # params dim: batch, beta params (2), arm
    action = np.zeros(N, dtype = np.int8)
    for i in range(N): # Loop through batch
        a0, b0 = params[i,:,0] # Arm 0
        a1, b1 = params[i,:,1] # Arm 1
            
	## Arm 0 Pulled
        # Difference in entropy when 0 (failure) observed
        dH0_0 = Beta_Entropy(a0, b0+1, a1, b1)
        # Difference in entropy when 1 (success) observed
        dH0_1 = Beta_Entropy(a0+1, b0, a1, b1)
        # Expected decrease in entropy
        dH0 = Beta_Post(False,a0,b0)*dH0_0 + Beta_Post(True,a0,b0)*dH0_1
                
        ## Arm 1 Pulled
        # Difference in entropy when 0 (failure) observed
        dH1_0 = Beta_Entropy(a0, b0, a1, b1+1)
        # Difference in entropy when 1 (success) observed
        dH1_1 = Beta_Entropy(a0, b0, a1+1, b1)
        # Expected decrease in entropy
        dH1 = Beta_Post(False,a1,b1)*dH1_0 + Beta_Post(True,a1,b1)*dH1_1
                
        action[i] = int(dH0 > dH1) # pick action that decreases H more (i.e. dH more negative)
    return action



def main(args):
    # Input unpacking
    p1 = args.p1            # Success probability of the first arm
    p2 = args.p2            # Success probability of the second arm
    p = np.array([p1,p2])   # Prob. of arms
    n = args.n              # Total number of plays
    N = args.N              # Batch/replicas (for averaging)
    n_rec = args.n_rec      # Record every n plays
    algo = args.algo        # 'Thompson', 'Infomax'
    seed = args.seed        # Random seed
    
    # Set random seed
    np.random.seed(seed)
        
    # Output initialization
    curr_cum_reward = np.zeros(N)
    curr_cum_subopt = np.zeros(N) # Number of times you pull suboptimal arm
    cum_reward = np.zeros([N,int(n/n_rec)]) # Record cumulative reward at each step
    cum_subopt = np.zeros([N,int(n/n_rec)]) # Record number of suboptimal plays
    plays = np.zeros(int(n/n_rec))
    
    # Setup Algorithm
    if algo == 'Beta':
        beta_params = np.ones([N,2,2])
    #elif algo == 'Gauss':
    
    # Simulation
    t_start = time.time()
    t1 = time.time()
    for t in range(n):
    
        # Choose action
        if algo == 'Beta':
            action = Beta_Action(beta_params, N)
	
        # Record suboptimal plays
        curr_cum_subopt += action # Default Arm 1 is suboptimal
        if t % n_rec == 0:
            cum_subopt[:,int(t/n_rec)] = curr_cum_subopt

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
        if algo == 'Beta':
            # Update a's
            beta_params[:,0][np.arange(N),action] += r
            # Update b's
            beta_params[:,1][np.arange(N),action] += 1-r
        
        # Time
        if args.v and (t+1) % int(n/10) == 0:
            t2 = time.time()
            print('Runtime for plays ' + str(t-(int(n/10)-1)) + ' - ' + str(t) + ': ' + str(t2-t1) + ' s')
            t1 = t2
                
    t_end = time.time()
        
    output = {'cum_reward':cum_reward, 'cum_subopt':cum_subopt, 'beta_params':beta_params, 'plays': plays}

    if args.v:
        print('Total runtime: ' + str(t_end-t_start) + ' s')
        print("Mean Total Reward:", np.mean(cum_reward[:,-1]) )
        print("Mean Suboptimal Plays:", np.mean(cum_subopt[:,-1]) )
        
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
    # Actual regret
    regret_act = c_mean * (p1 - p2)
    plt.plot(plays, regret_act, 'r', linewidth=3, label=algo)
    plt.legend(loc='best')
    plt.gca().set_xscale('log',basex=10)
    plt.ylim([0, np.ceil(np.amax(regret_act)) ]) # Adaptive max of graph
    plt.xlabel('Total number of plays')
    plt.ylabel(r'Regret $<n_2>(p_1-p_2)$')
    plt.title('Regret averaged over ' + str(N) + ' replicas')
    if savedata:
        savePath = 'Save/'
        save_time = datetime.datetime.today().strftime('_%Y-%m-%d_%H:%M_')
        pickle_name = name + save_time + "_regret_" + '.pkl'
        with open(savePath + pickle_name, 'wb') as f:
            pickle.dump(regret_act, f)
    if savefig:
        plt.savefig('regret_' + name + '.png')
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # Parse inputs ------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='2-armed bandit')
    
    parser.add_argument("--p1", default=0.9, type=float, help="Success probability of the superior arm")
    parser.add_argument("--p2", default=0.8, type=float, help="Success probability of the inferior arm")
    parser.add_argument("--n", default=1000, type=int, help="Total number of plays")
    parser.add_argument("--N", default=1, type=int, help="Replicas (For averaging)")
    parser.add_argument("--n_rec", default=1, type=int, help="Record every n plays")
    parser.add_argument("--algo", default='Beta', type=str, help="Reward probability model: 'Beta' or 'Gauss'")
    parser.add_argument("--seed", default=111, type=int, help="Random seed")
    parser.add_argument("--savefig", action = "store_true", help="Save figures")
    parser.add_argument("--savedata", action = "store_true", help="Save regret data")
    parser.add_argument("-v", action = "store_true", help="Print messages")
    
    args = parser.parse_args()

    # Run main function -------------------------------------------------------------------------
    output = main(args)
    
    # Save output -------------------------------------------------------------------------------

    filename = 'algo=' + str(args.algo)
    # Modify filename if nondefault args are used
    if args.p1 != 0.9:
        filename = filename + '_p1=' + str(args.p1)
    if args.p2 != 0.8:
        filename = filename + '_p2=' + str(args.p2)
    if args.n != 1000:
        filename = filename + '_n=' + str(args.n)
    if args.N != 1:
        filename = filename + '_N=' + str(args.N)
    if args.n_rec != 1:
        filename = filename + '_n_rec=' + str(args.n_rec)
    if args.seed != 111:
        filename = filename + '_seed=' + str(args.seed)
                
    #pickle.dump(output, open(name + '.pickle',"wb"))
    
    # Extract outputs ---------------------------------------------------------------------------
    a = output['cum_reward']
    b = output['beta_params']
    c = output['cum_subopt']
    plays = output['plays']
    c_mean = np.mean(c,0) # Take average over batches
    
    # Plot beta distribution --------------------------------------------------------------------
    #x = np.linspace(0,1,1000) # Parameter space [0,1]
    #plot_beta(x,b[0,0,0],b[0,0,1],b[0,1,0],b[0,1,1], args.p1, args.p2, args.n, filename, args.savefig)
    
    # Plot regret -------------------------------------------------------------------------------
    plot_regret(plays, c_mean, args.p1, args.p2, args.N, filename, args.savefig, args.savedata, args.algo)
