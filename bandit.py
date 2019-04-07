import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

import time
import argparse
import pickle

# KL-divergence
def KLdivergence(p,q):
    return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

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
    curr_cum_subopt = np.zeros(N)
    cum_reward = np.zeros([N,int(n/n_rec)]) # Record cumulative reward at each step
    cum_subopt = np.zeros([N,int(n/n_rec)]) # Record number of suboptimal plays
    plays = np.zeros(int(n/n_rec))
    
    # Setup
    if algo == 'Thompson':
        # Initialize beta distribution parameters
        # beta_params_(batch, {alpha,beta}, arm)
        # 0.5 = uninformative prior, 1 = uniform prior
        beta_params = 0.5*np.ones([N,2,2]) # [[[a1,a2],[b1,b2]], ...]
    
    # Simulation
    t_start = time.time()
    t1 = time.time()
    for t in range(n):
    
        # Choose arm
        if algo == 'Thompson':
            theta = np.random.beta(beta_params[:,0,:],beta_params[:,1,:]) # [batch, arm]
            action = np.argmax(theta,1) # Choose the arm corresponding to the optimal draw
            
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
        
        #if t > 0:
        #    cum_reward[:,t] = cum_reward[:,t-1]
        #cum_reward[:,t] += r
            
        # Update parameters
        if algo == 'Thompson':
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
    if verbose:
        print('Total runtime: ' + str(t_end-t_start) + ' s')
    
    output = {'cum_reward':cum_reward, 'cum_subopt':cum_subopt, 'beta_params':beta_params, 'plays': plays}
    
    return output

if __name__ == "__main__":
    # Parse inputs ------------------------------------------------------------------------------
    # Example command: python bandit.py --n 1000000 --N 100 --verbose 1
    parser = argparse.ArgumentParser(description='2-armed bandit')
    
    parser.add_argument("--p1", default=0.9, type=float, help="Success probability of the superior arm")
    parser.add_argument("--p2", default=0.8, type=float, help="Success probability of the inferior arm")
    parser.add_argument("--n", default=1000, type=int, help="Total number of plays")
    parser.add_argument("--N", default=1, type=int, help="Replicas (For averaging)")
    parser.add_argument("--n_rec", default=1, type=int, help="Record every n plays")
    parser.add_argument("--algo", default='Thompson', type=str, help="Decision algorithm; 'Thompson', 'Infomax'")
    parser.add_argument("--seed", default=111, type=int, help="Random seed")
    parser.add_argument("--savefig", default=0, type=int, help="Save figures")
    parser.add_argument("--verbose", default=0, type=int, help="Print messages")
    
    args = parser.parse_args()

    # Run main function -------------------------------------------------------------------------
    output = main(args)
    
    # Outputs
    a = output['cum_reward']
    b = output['beta_params']
    c = output['cum_subopt']
    plays = output['plays']
    c_mean = np.mean(c,0) # Take average over batches
    
    # Save outputs
    name =  'p1=' + str(args.p1) + \
            '_p2=' + str(args.p2) + \
            '_n=' + str(args.n) + \
            '_N=' + str(args.N) + \
            '_n_rec=' + str(args.n_rec) + \
            '_algo=' + str(args.algo) + \
            '_seed=' + str(args.seed)
                
    pickle.dump(output, open(name + '.pickle',"wb"))
    
    # Plot beta distribution --------------------------------------------------------------------
    plt.figure()
    x = np.linspace(0,1,1000)
    beta1 = beta.pdf(x,b[0,0,0],b[0,1,0])
    beta2 = beta.pdf(x,b[0,0,1],b[0,1,1])
    plt.plot(x, beta1, 'r', linewidth=3, label='Superior arm')
    plt.plot(x, beta2, 'b', linewidth=3, label='Inferior arm')
    plt.axvline(x=args.p1,color='r', linewidth=1, linestyle='--')
    plt.axvline(x=args.p2,color='b', linewidth=1, linestyle='--')
    plt.legend(loc='best')
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$p(\theta)$')
    plt.title('Learned beta distributions for each arm after ' + str(args.n) + ' plays')
    if args.savefig:
        plt.savefig('beta_distribution_' + name + '.png')
        plt.close()
    else:
        plt.show()
    
    # Plot regret -------------------------------------------------------------------------------
    plt.figure()
    # Optimal expected number of suboptimal plays (Lai-Robbins lower bound)
    LB = np.log(plays)/KLdivergence(args.p1,args.p2)
    # Optimal regret
    regret_opt = np.maximum(LB * (args.p1 - args.p2)-20, np.zeros(LB.shape))
    # Actual regret
    regret_act = c_mean * (args.p1 - args.p2)
    plt.plot(plays, regret_opt, 'k', linewidth=3, label='Lai-Robbins bound slope')
    plt.plot(plays, regret_act, 'r', linewidth=3, label='Thompson sampling')
    plt.legend(loc='best')
    plt.gca().set_xscale('log',basex=10)
    plt.ylim([0,30])
    plt.xlabel('Total number of plays')
    plt.ylabel(r'Regret $<n_2>(p_1-p_2)$')
    plt.title('Regret averaged over ' + str(args.N) + ' replicas')
    if args.savefig:
        plt.savefig('regret_' + name + '.png')
        plt.close()
    else:
        plt.show()