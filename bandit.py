import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

import time
import argparse

# KL-divergence
def KLdivergence(p,q):
    return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

def main(args):
    # Input unpacking
    p1 = args.p1            # Success probability of the first arm
    p2 = args.p2            # Success probability of the second arm
    n = args.n              # Total number of plays
    N = args.N              # Batch/replicas (for averaging)
    algo = args.algo        # 'Thompson', 'Infomax'
    verbose = args.verbose  # Print messages
    
    # Input representation
    p = [p1,p2]             # Prob. of arms
    
    # Output initialization
    cum_reward = np.zeros(n) # Record cumulative reward at each step
    cum_subopt = np.zeros(n) # Record number of suboptimal plays
    
    # Setup
    if algo == 'Thompson':
        # Initialize beta distribution parameters
        # beta_params_(batch, {alpha,beta}, arm)
        beta_params = 0.5*np.ones([2,2]) # [[[a1,a2],[b1,b2]], ...], uninformative prior = 0.5
    
    # Simulation
    t_start = time.time()
    t1 = time.time()
    for t in range(n):
    
        # Choose arm
        if algo == 'Thompson':
            theta = np.random.beta(beta_params[0,:],beta_params[1,:])
            action = np.argmax(theta)
        # Record suboptimal plays
        if t > 0:
            cum_subopt[t] = cum_subopt[t-1]
        cum_subopt[t] += int(action)

        # Play the arm
        chance = np.random.random() # Random number between 0 and 1
        r = int(p[action] > chance)         # Get reward
        
        # Record cumulative rewards
        if t > 0:
            cum_reward[t] = cum_reward[t-1]
        cum_reward[t] += r
            
        # Update parameters
        if algo == 'Thompson':
            beta_params[:,action] += np.array([r,1-r])
        
        # Time
        if verbose:
            if (t+1) % int(n/10) == 0:
                t2 = time.time()
                print('Runtime for plays ' + str(t-(int(n/10)-1)) + ' - ' + str(t) + ': ' + str(t2-t1) + ' s')
                t1 = t2
                
    t_end = time.time()
    print('Total runtime: ' + str(t_end-t_start) + ' s')
    
    output = {'cum_reward':cum_reward, 'cum_subopt':cum_subopt, 'beta_params':beta_params}
    
    return output

if __name__ == "__main__":
    # Parse inputs ------------------------------------------------------------------------------
    # Example command: python bandit.py --p1 0.95 --n 500 --algo Infomax
    parser = argparse.ArgumentParser(description='2-armed bandit')
    
    parser.add_argument("--p1", default=0.9, type=float, help="Success probability of the superior arm")
    parser.add_argument("--p2", default=0.8, type=float, help="Success probability of the inferior arm")
    parser.add_argument("--n", default=1000, type=int, help="Total number of plays")
    parser.add_argument("--N", default=1, type=int, help="Replicas (For averaging)")
    parser.add_argument("--algo", default='Thompson', type=str, help="Decision algorithm; 'Thompson', 'Infomax'")
    parser.add_argument("--verbose", default=0, type=str, help="Print messages")
    
    args = parser.parse_args()

    # Run main function -------------------------------------------------------------------------
    output = main(args)
    
    # Outputs
    a = output['cum_reward']
    b = output['beta_params']
    c = output['cum_subopt']
    
    # Plot beta distribution --------------------------------------------------------------------
    plt.figure()
    x = np.linspace(0,1,1000)
    beta1 = beta.pdf(x,b[0,0],b[1,0])
    beta2 = beta.pdf(x,b[0,1],b[1,1])
    plt.plot(x, beta1, 'r', linewidth=3, label='Superior arm')
    plt.plot(x, beta2, 'b', linewidth=3, label='Inferior arm')
    plt.axvline(x=args.p1,color='r', linewidth=1, linestyle='--')
    plt.axvline(x=args.p2,color='b', linewidth=1, linestyle='--')
    plt.legend(loc='best')
    plt.show()
    
    # Plot regret -------------------------------------------------------------------------------
    plt.figure()
    num_plays = len(a)
    plays = np.arange(1,num_plays+1)
    # Optimal expected number of suboptimal plays (Lai-Robbins lower bound)
    LB = np.log(plays)/KLdivergence(args.p1,args.p2)
    # Optimal regret
    regret_opt = np.maximum(LB * (args.p1 - args.p2)-20, np.zeros(LB.shape))
    # Actual regret
    regret_act = c * (args.p1 - args.p2)
    plt.plot(plays, regret_opt, 'k', linewidth=3, label='Lai-Robbins optimal regret')
    plt.plot(plays, regret_act, 'r', linewidth=3, label='Thompson sampling')
    plt.legend(loc='best')
    plt.gca().set_xscale('log',basex=10)
    plt.show()