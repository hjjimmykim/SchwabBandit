# Two states s0 and s1; start at s0 initially
# At s0, the actions available are a0 = lever w/ 0.8 prob., a1 = switch to s1
# At s0, the actions available are a0 = lever w/ 0.9 prob., a1 = switch to s0

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.integrate import quad

import time
import argparse
import pickle

# Markov Decision Process Functions -------------------------------------------------------------

# Optimal Q-values
def Q_opt(p0, p1, gamma):
    gamma_c = p0/p1 # Critical gamma
    
    Q_opt = np.zeros([2,2]) # [s, a]
    
    if gamma >= gamma_c:
        Q_opt[0,0] = p0 + gamma**2 * p1/(1-gamma)
    else:
        Q_opt[0,0] = p0/(1-gamma)
    Q_opt[0,1] = gamma*p1/(1-gamma)
    Q_opt[1,0] = p1/(1-gamma)
    if gamma >= gamma_c:
        Q_opt[1,1] = gamma**2 * p1/(1-gamma)
    else:
        Q_opt[1,1] = gamma * p0/(1-gamma)
        
    return Q_opt
    
# Reinforcement learning functions --------------------------------------------------------------

# Initialize Q-values
def initialize(num_states, num_actions, init_scheme = 'Zero'):
    if init_scheme == 'Zero':
        Q = np.zeros([num_states, num_actions])
    elif init_scheme == 'Random':
        Q = np.random.randn(num_states, num_actions)
    else:
        raise Exception('Initialization scheme unknown')
    
    return Q

# Information Theory Functions ------------------------------------------------------------------

# KL-divergence
def KLdivergence(p,q):
    # Takes two probability numbers and returns a number
    
    return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))
    
# Probability distribution over max probability
def Rhomax(x, a1, a2, b1, b2):
    # Takes numbers and returns a prob. number
    
    pdf1 = beta.pdf(x,a1,b1)
    pdf2 = beta.pdf(x,a2,b2)
    cdf1 = beta.cdf(x,a1,b1)
    cdf2 = beta.cdf(x,a2,b2)
    
    rho = pdf1 * cdf2 + pdf2 * cdf1
    
    return rho

# Vectorized version of above (probably not useful)
def Rhomax_vectorized(x, beta_params):
    # Takes full beta_params and returns an array (over batch)
    
    a1 = beta_params[:,0,0] # alpha for arm 0
    b1 = beta_params[:,1,0] # beta for arm 0
    a2 = beta_params[:,0,1] # alpha for arm 1
    b2 = beta_params[:,1,1] # beta for arm 1
    
    pdf1 = beta.pdf(x,a1,b1)
    pdf2 = beta.pdf(x,a2,b2)
    cdf1 = beta.cdf(x,a1,b1)
    cdf2 = beta.cdf(x,a2,b2)
    
    rho = pdf1 * cdf2 + pdf2 * cdf1
    
    return rho
    
# Posterior distribution over outcome given posterior over probs.
def Prob(x, a, b):
    # Takes x = 0 or 1 and a,b = numbers; returns a prob. number
    
    if x == 1: # Success
        integrand = lambda p: beta.pdf(p,a,b) * p
    elif x == 0:
        integrand = lambda p: beta.pdf(p,a,b) * (1-p)
        
    integral = quad(integrand, 0, 1)[0]
    
    return integral
    
# Differential entropy over max probability
def Entropy(a1, a2, b1, b2):
    # Takes numbers and returns a number
    
    integrand = lambda p: -Rhomax(p, a1, a2, b1, b2) * np.log(Rhomax(p, a1, a2, b1, b2))
    
    integral = quad(integrand, 0, 1)[0]
    
    return integral

# Main function ---------------------------------------------------------------------------------

def main(args):
    # Input unpacking
    p0 = args.p0            # Success probability of the first arm
    p1 = args.p1            # Success probability of the second arm
    n = args.n              # Total number of plays
    N = args.N              # Batch/replicas (for averaging)
    n_rec = args.n_rec      # Record every n plays
    algo = args.algo        # 'Thompson', 'Infomax'
    seed = args.seed        # Random seed
    verbose = args.verbose  # Print messages (time)
    
    # Set random seed
    np.random.seed(seed)
    
    # Variable representations
    state = 0                         # Current state (0 or 1)
    p = np.array([p0,p1])             # Prob. of arms

    # Output initialization
    curr_cum_reward = np.zeros(N)
    curr_cum_subopt = np.zeros(N)
    cum_reward = np.zeros([N,int(n/n_rec)]) # Record cumulative reward at each step
    cum_subopt = np.zeros([N,int(n/n_rec)]) # Record number of non-optimal plays (optimal = moving to s1 and playing the arm)
    plays = np.zeros(int(n/n_rec))
    
    # Setup
    if algo == 'Qlearn':
        Q = initialize(2,2)
    elif algo == 'Infomax':
        # Initialize beta distribution parameters
        # beta_params_(batch, {alpha,beta}, arm)
        # 0.5 = uninformative prior, 1 = uniform prior
        pass
        '''
        beta_params = 1*np.ones([N,2,2])
        '''
    
    # Simulation
    t_start = time.time()
    t1 = time.time()
    for t in range(n):
        # Choose arm
        if algo == 'Qlearn':
            epsilon = 0.05
            if np.random.random() > epsilon: # Pick optimal
                action = np.argmax(Q[state])
            else:                            # Pick random
                action = np.random.randint(0,2)
        elif algo == 'Infomax':
            action = np.zeros(N, dtype = np.int8)
            '''
            for i in range(N): # Loop through batch
                a1, b1 = beta_params[i,:,0] # Arm 0
                a2, b2 = beta_params[i,:,1] # Arm 1
                
                # Arm 0
                # Difference in entropy when 0 (failure) observed
                delH0_0 = Entropy(a1, a2, b1 + 1, b2) - Entropy(a1, a2, b1, b2)
                # Difference in entropy when 1 (success) observed
                delH0_1 = Entropy(a1 + 1, a2, b1, b2) - Entropy(a1, a2, b1, b2)
                # Expected decrease in entropy
                dH0 = Prob(0,a1,b1)*delH0_0 + Prob(1,a1,b1)*delH0_1
                
                # Arm 1
                # Difference in entropy when 0 (failure) observed
                delH1_0 = Entropy(a1, a2, b1, b2 + 1) - Entropy(a1, a2, b1, b2)
                # Difference in entropy when 1 (success) observed
                delH1_1 = Entropy(a1, a2 + 1, b1, b2) - Entropy(a1, a2, b1, b2)
                # Expected decrease in entropy
                dH1 = Prob(0,a2,b2)*delH1_0 + Prob(1,a2,b2)*delH1_1
                
                action[i] = int(dH0 > dH1) # pick action that decreases H more (i.e. dH more negative)
            '''
            
        # Record suboptimal plays
        curr_cum_subopt += int(state==action) # s=0,a=0 => play suboptimal arm s=1,a=1 => move to suboptimal arm
        if t % n_rec == 0:
            cum_subopt[:,int(t/n_rec)] = curr_cum_subopt

        # Play
        if action == 0: # Arm played
            chance = np.random.random() # Random number between 0 and 1
            r = (p[state] > chance).astype(int)         # Get reward
            state_new = state
        else:           # State changed
            state_new = int(not(state))
            r = 0
        
        # Keep track of cumulative rewards
        curr_cum_reward += r
        # Record cumulative rewards
        if t % n_rec == 0:
            cum_reward[:,int(t/n_rec)] = curr_cum_reward
            plays[int(t/n_rec)] = t+1
            
        # Update parameters
        if algo == 'Qlearn':
            alpha = 0.01
            gamma = 0.8
            Q[state][action] = (1-alpha)*Q[state][action] + alpha*(r + gamma*max(Q[state_new]))
        elif algo == 'Infomax':
            pass
            '''
            # Update a
            beta_params[:,0,:][np.arange(N),action] += r
            # Update b
            beta_params[:,1,:][np.arange(N),action] += 1-r
            '''
            
        # Move to next state
        state = state_new
        
        # Time
        if verbose:
            if (t+1) % int(n/10) == 0:
                t2 = time.time()
                print('Runtime for plays ' + str(t-(int(n/10)-1)) + ' - ' + str(t) + ': ' + str(t2-t1) + ' s')
                t1 = t2
                
    t_end = time.time()
    if verbose:
        print('Total runtime: ' + str(t_end-t_start) + ' s')
    
    output = {'cum_reward':cum_reward, 'cum_subopt':cum_subopt, 'Q':Q, 'plays': plays}
    
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
def plot_regret(plays, c_mean, p1, p2, N, name, savefig):
    plt.figure()
    # Optimal expected number of suboptimal plays (Lai-Robbins lower bound)
    LB = np.log(plays)/KLdivergence(p1,p2)
    # Optimal regret
    regret_opt = np.maximum(LB * (p1 - p2)-20, np.zeros(LB.shape))
    # Actual regret
    regret_act = c_mean * (p1 - p2)
    plt.plot(plays, regret_opt, 'k', linewidth=3, label='Lai-Robbins bound slope')
    plt.plot(plays, regret_act, 'r', linewidth=3, label='Thompson sampling')
    plt.legend(loc='best')
    plt.gca().set_xscale('log',basex=10)
    plt.ylim([0,30])
    plt.xlabel('Total number of plays')
    plt.ylabel(r'Regret $<n_2>(p_1-p_2)$')
    plt.title('Regret averaged over ' + str(N) + ' replicas')
    if savefig:
        plt.savefig('regret_' + name + '.png')
        plt.close()
    else:
        plt.show()
        
# Plot reward
def plot_reward(plays, cum_reward, p1, name, savefig):
    plt.figure()
    # Optimal reward
    cum_reward_opt = p1 * plays - p1 # First optimal action is just transitioning
    plt.plot(plays, cum_reward_opt, 'k', linewidth=3, label='Optimal')
    plt.plot(plays, cum_reward[0], 'r', linewidth=3, label='Q-learning')
    plt.legend(loc='best')
    #plt.gca().set_xscale('log',basex=10)
    plt.ylim([0,max(cum_reward_opt)*1.5])
    plt.xlabel('Total number of plays')
    plt.ylabel('Reward')
    plt.title('Cumulative reward')
    if savefig:
        plt.savefig('Multistate_reward_' + name + '.png')
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # Parse inputs ------------------------------------------------------------------------------
    # Example command: python bandit.py --n 1000000 --N 100 --verbose 1
    parser = argparse.ArgumentParser(description='2-armed bandit')
    
    parser.add_argument("--p0", default=0.8, type=float, help="Success probability of the s0 arm")
    parser.add_argument("--p1", default=0.9, type=float, help="Success probability of the s1 arm")
    parser.add_argument("--n", default=1000, type=int, help="Total number of turns")
    parser.add_argument("--N", default=1, type=int, help="Replicas (For averaging)")
    parser.add_argument("--n_rec", default=1, type=int, help="Record every n turns")
    parser.add_argument("--algo", default='Qlearn', type=str, help="Decision algorithm; 'Qlearn', 'Infomax'")
    parser.add_argument("--seed", default=111, type=int, help="Random seed")
    parser.add_argument("--savefig", default=0, type=int, help="Save figures")
    parser.add_argument("--verbose", default=0, type=int, help="Print messages")
    
    args = parser.parse_args()

    # Run main function -------------------------------------------------------------------------
    output = main(args)
    
    # Save output -------------------------------------------------------------------------------

    name =  'p0=' + str(args.p0) + \
            '_p1=' + str(args.p1) + \
            '_n=' + str(args.n) + \
            '_N=' + str(args.N) + \
            '_n_rec=' + str(args.n_rec) + \
            '_algo=' + str(args.algo) + \
            '_seed=' + str(args.seed)
                
    #pickle.dump(output, open(name + '.pickle',"wb"))
    
    # Extract outputs ---------------------------------------------------------------------------
    cum_reward = output['cum_reward']
    Q = output['Q']
    cum_subopt = output['cum_subopt']
    plays = output['plays']

    # Plot beta distribution --------------------------------------------------------------------
    #x = np.linspace(0,1,1000) # Parameter space [0,1]
    #plot_beta(x,b[0,0,0],b[0,0,1],b[0,1,0],b[0,1,1],args.p1,args.p2,args.n,name,args.savefig)

    # Plot regret -------------------------------------------------------------------------------
    plot_reward(plays,cum_reward,args.p1,name,args.savefig)