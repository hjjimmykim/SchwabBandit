import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
#from scipy.stats import beta

#from scipy.integrate import quad
import ghalton

import time
import datetime
import argparse
import pickle


## Gaussian Process --------------------
def kernel(x, y, params):
# kernel function
    return params[0] * np.exp( -0.5 * params[1] * np.subtract.outer(x, y)**2 )

def conditional(x_new, x, y, params):
# Prob of test data conditioned on training data
    A = kernel(x_new, x_new, params)
    B = kernel(x_new, x, params)
    C = kernel(x, x, params)

    mu = np.linalg.inv(C).dot(B.T).T.dot(y)
    sigma = A - B.dot(np.linalg.inv(C).dot(B.T) )

    return( mu.squeeze(), sigma.squeeze() )

def predict(x, data, params, sigma, t):
# return predicted value and its std
    k = [ kernel(x, y, params) for y in data ]
    sig_inv = np.linalg.inv(sigma)
    y_pred = np.dot(k, sig_inv).dot(t)
    sigma_new = kernel(x, x, params) - np.dot(k, sig_inv).dot(k)
    return y_pred, sigma_new


# Entropy calculation (quasi-random Monte Carlo)
seq = ghalton.Halton(1)
def Entropy_est(a1, a2, b1, b2, samples=100):
    seq.reset()
    x = seq.get(samples)
    x = np.reshape(x, samples)
    integral = np.mean( rhomax_integrand(x,a1,a2,b1,b2) )
    return integral

def rho_max(state, mem0, mem1, params, num_samp=100):
    # memi = past sets of (mu0, mu1, r)
    seq.reset()
    r_samp = seq.get(num_samp)
    r_samp = np.reshape(r_samp, num_samp)
    # Arm 0
    sigma0 = kernel(mem0[:][0,2], mem0[:][0,2], params) # covariance of arm 0 training data
    P0 = predict(state, mem0[:][0,2], params, mem0[:][-1] )
    Intg1 = np.mean(predict(state, mem1[:]) )


# Main function ---------------------------------------------------------------------------------

def main(args):
    # Input unpacking
    p1 = args.p1            # Success probability of the first arm
    p2 = args.p2            # Success probability of the second arm
    n = args.n              # Total number of plays
    N = args.N              # Batch/replicas (for averaging)
    n_rec = args.n_rec      # Record every n plays
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
    
    # Simulation
    t_start = time.time()
    t1 = time.time()

    state_mem = np.array([])

    for t in range(n):
    
        # Choose arm
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
    parser.add_argument("--seed", default=111, type=int, help="Random seed")
    parser.add_argument("--savefig", default=0, type=int, help="Save figures")
    parser.add_argument("--savedata", default=0, type=int, help="Save regret data")
    parser.add_argument("--verbose", default=0, type=int, help="Print messages")
    
    args = parser.parse_args()

    algo == 'GP-Infomax'

    # Run main function -------------------------------------------------------------------------
    output = main(args)
    
    # Save output -------------------------------------------------------------------------------

    name =  'p1=' + str(args.p1) + \
            '_p2=' + str(args.p2) + \
            '_n=' + str(args.n) + \
            '_N=' + str(args.N) + \
            '_n_rec=' + str(args.n_rec) + \
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
    plot_regret(plays,c_mean,args.p1,args.p2,args.N,name,args.savefig, args.savedata, algo)
