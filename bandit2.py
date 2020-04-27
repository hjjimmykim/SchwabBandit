import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.stats import beta, norm, cauchy

import ghalton

import time
import datetime
import argparse
import pickle


## Estimate rhomax entropy via sampling
seq = ghalton.Halton(1)
MC_samples = 100    # default number of samples for MC integration
def Beta_Entropy(w0, l0, w1, l1):
    global MC_samples
    seq.reset()
    x = seq.get(MC_samples)
    x = np.reshape(x, MC_samples)

    pdf0 = beta.pdf(x, w0+1, l0+1)
    pdf1 = beta.pdf(x, w1+1, l1+1)
    cdf0 = beta.cdf(x, w0+1, l0+1)
    cdf1 = beta.cdf(x, w1+1, l1+1)
    
    rho = pdf0 * cdf1 + pdf1 * cdf0
    integral = np.mean( -rho * np.log( rho ) )

    return integral

def Gauss_Entropy(w0, l0, w1, l1):
    mu0 = (w0 + 1)/(w0 + l0 + 2)    # Expectation of beta
    sig0 = np.sqrt( (w0+1)*(l0+1)/( (w0+l0+2)**2 * (w0+l0+3) ) )    # Std of beta
    mu1 = (w1 + 1)/(w1 + l1 + 2)
    sig1 = np.sqrt( (w1+1)*(l1+1)/( (w1+l1+2)**2 * (w1+l1+3) ) )

    global MC_samples
    seq.reset()
    x = seq.get(MC_samples)
    x = np.reshape(x, MC_samples)

    pdf0 = norm.pdf(x, mu0, sig0)
    pdf1 = norm.pdf(x, mu1, sig1)
    cdf0 = norm.cdf(x, mu0, sig0)
    cdf1 = norm.cdf(x, mu1, sig1)

    rho = pdf0 * cdf1 + pdf1 * cdf0
    integral = np.mean( -rho * np.log( rho ) )

    return integral

def Cauchy_Entropy(w0, l0, w1, l1):
    if w0 == 0 and l0 == 0:
        x0 = 0.5
    else:
        x0 = w0/(w0 + l0)    # Mode of beta
    sig0 = np.sqrt( (w0+1)*(l0+1)/( (w0+l0+2)**2 * (w0+l0+3) ) )    # Std of beta
    if w1 == 0 and l1 ==0:
        x1 = 0.5
    else:
        x1 = w1/(w1 + l1)
    sig1 = np.sqrt( (w1+1)*(l1+1)/( (w1+l1+2)**2 * (w1+l1+3) ) )

    global MC_samples
    seq.reset()
    x = seq.get(MC_samples)
    x = np.reshape(x, MC_samples)

    pdf0 = cauchy.pdf(x, x0, sig0)
    pdf1 = cauchy.pdf(x, x1, sig1)
    cdf0 = cauchy.cdf(x, x0, sig0)
    cdf1 = cauchy.cdf(x, x1, sig1)
    #norm0 = 1/(cauchy.cdf(1,x0,sig0) - cauchy.cdf(0,x0,sig0) )
    #norm1 = 1/(cauchy.cdf(1,x1,sig1) - cauchy.cdf(0,x1,sig1) )

    rho = pdf0 * cdf1 + pdf1 * cdf0#norm0 * norm1 * (pdf0 * cdf1 + pdf1 * cdf0)
    integral = np.mean( -rho * np.log( rho ) )

    return integral


## Evidence functions
def Beta_P(outcome, w, l):
    if outcome:
        return (w+1)/(w+l+2)
    else:
        return (l+1)/(w+l+2)

def Gauss_P(outcome, w, l):
    mu = (w + 1)/(w + l + 2)
    sig = np.sqrt( (w+1)*(l+1)/( (w+l+2)**2 * (w+l+3) ) )

    global MC_samples
    seq.reset()
    p = seq.get(MC_samples)
    p = np.reshape(p, MC_samples)

    likelihood = p**outcome * (1-p)**(1-outcome)
    prior = norm.pdf(p, mu, sig)
    evidence = np.mean( likelihood * prior )

    return evidence

def Cauchy_P(outcome, w, l):
    if w == 0 and l == 0:
        x = 0.5
    else:
        x = w/(w + l)
    sig = np.sqrt( (w+1)*(l+1)/( (w+l+2)**2 * (w+l+3) ) )

    global MC_samples
    seq.reset()
    p = seq.get(MC_samples)
    p = np.reshape(p, MC_samples)

    likelihood = p**outcome * (1-p)**(1-outcome)
    prior = cauchy.pdf(p, x, sig)#/(cauchy.cdf(1,x,sig) - cauchy.cdf(0,x,sig) )
    evidence = np.mean( likelihood * prior )

    return evidence


def choose_arm(results, N, algo):
    if algo == 'Beta':
        Entropy = Beta_Entropy
        Evidence = Beta_P
    elif algo == 'Gauss':
        Entropy = Gauss_Entropy
        Evidence = Gauss_P
    else:
        Entropy = Cauchy_Entropy
        Evidence = Cauchy_P

    action = np.zeros(N, dtype = np.int8)
    
    for i in range(N): # Loop through batch
        w0, l0 = results[i,:,0] # Arm 0
        w1, l1 = results[i,:,1] # Arm 1

	    ## Arm 0 Pulled
        # 0 (failure) observed
        H0_0 = Entropy(w0, l0+1, w1, l1)
        # 1 (success) observed
        H0_1 = Entropy(w0+1, l0, w1, l1)
        # Expected decrease in entropy
        dH0 = Evidence(0, w0, l0) * H0_0 + Evidence(1, w0, l0) * H0_1
                
        ## Arm 1 Pulled
        # 0 (failure) observed
        H1_0 = Entropy(w0, l0, w1, l1+1)
        # 1 (success) observed
        H1_1 = Entropy(w0, l0, w1+1, l1)
        # Expected decrease in entropy
        dH1 = Evidence(0, w1, l1) * H1_0 + Evidence(1, w1, l1) * H1_1
                
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
    
    # Setup Parameters (dim: batch, results, arm)
    results_array = np.zeros([N,2,2])
    
    # Simulation
    t_start = time.time()
    t1 = time.time()
    for t in range(n):
    
        # Choose action
        action = choose_arm(results_array, N, algo)
	
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
        # Update wins
        results_array[:,0][np.arange(N),action] += r
        # Update losses
        results_array[:,1][np.arange(N),action] += 1-r
        
        # Time
        if args.v and (t+1) % int(n/10) == 0:
            t2 = time.time()
            print('Runtime for plays ' + str(t-(int(n/10)-1)) + ' - ' + str(t) + ': ' + str(t2-t1) + ' s')
            t1 = t2
                
    t_end = time.time()
        
    output = {'cum_reward':cum_reward, 'cum_subopt':cum_subopt, 'results_array':results_array, 'plays': plays}

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
    parser.add_argument("--int_samp", default=100, type=int, help="Number of samples for MC integration")
    parser.add_argument("--algo", default='Beta', type=str, help="Reward probability model: 'Beta' or 'Gauss'")
    parser.add_argument("--seed", default=111, type=int, help="Random seed")
    parser.add_argument("--savefig", action = "store_true", help="Save figures")
    parser.add_argument("--savedata", action = "store_true", help="Save regret data")
    parser.add_argument("-v", action = "store_true", help="Print messages")
    
    args = parser.parse_args()
    MC_samples = args.int_samp

    # Run main function -------------------------------------------------------------------------
    print("Algorithm:", args.algo)
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
    b = output['results_array']
    c = output['cum_subopt']
    plays = output['plays']
    c_mean = np.mean(c,0) # Take average over batches
    
    # Plot beta distribution --------------------------------------------------------------------
    #x = np.linspace(0,1,1000) # Parameter space [0,1]
    #plot_beta(x,b[0,0,0],b[0,0,1],b[0,1,0],b[0,1,1], args.p1, args.p2, args.n, filename, args.savefig)
    
    # Plot regret -------------------------------------------------------------------------------
    plot_regret(plays, c_mean, args.p1, args.p2, args.N, filename, args.savefig, args.savedata, args.algo)
