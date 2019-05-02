# Chain
# s0 -> s1 -> s2 -> s3 -> s4 -> s5
# Can return to s0 from any state

# N. Meuleau and P. Bourgine. Exploration of multi-states environments: Local measures and back-propogation of uncertainty (1998)
# Six states, two actions: return to s0 or move to right.
# Moving right at s5 gives 10 reward.
# Returning to s0 gives 2 reward.
# gamma = 0.99 => optimal to move right everywhere

# R. Dearden, N. Friedman, S. Russell. Bayesian Q-learning (1998)
# With prob. 0.2, opposite action is performed

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.stats import t as tdist
from scipy.integrate import quad

import time
import argparse
import pickle

# Reinforcement learning functions --------------------------------------------------------------

# Initialize Q-values
def initialize(num_replicas, num_states, num_actions, init_scheme = 'Zero'):
    if init_scheme == 'Zero':
        Q = np.zeros([num_replicas, num_states, num_actions])
    elif init_scheme == 'Random':
        Q = np.random.randn(num_replicas, num_states, num_actions)
    else:
        raise Exception('Initialization scheme unknown')
    
    return Q

# Statistics Functions --------------------------------------------------------------------------

# Sample mu & tau from normal-gamma distribution
def sample_NG(params):
    mu0, lamb, alpha, beta = params
    
    # Sample tau (shape, rate)
    tau_sample = np.random.gamma(alpha, 1/beta)

    # Sample mu (mean, std)
    mu_sample = np.random.normal(mu0,np.sqrt(1/(lamb*tau_sample)))
    
    return mu_sample, tau_sample
    
# Probability distribution over max probability
def Rhomax(x, a1, a2, b1, b2):
    # Takes numbers and returns a prob. number
    
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
    gamma = args.gamma      # Discount factor
    n = args.n              # Total number of plays
    N = args.N              # Batch/replicas (for averaging)
    n_rec = args.n_rec      # Record every n plays
    algo = args.algo        # 'Thompson', 'Infomax'
    seed = args.seed        # Random seed
    verbose = args.verbose  # Print messages (time)
    
    # Set random seed
    np.random.seed(seed)
    
    # Variable representations
    state = np.zeros(N,dtype=int)                         # Current state (0-5)

    # Output initialization
    curr_cum_reward = np.zeros(N)
    curr_cum_subopt = np.zeros(N)
    cum_reward = np.zeros([N,int(n/n_rec)]) # Record cumulative reward at each step
    plays = np.zeros(int(n/n_rec))
    
    # Setup
    if algo == 'Qlearn':
        Q = initialize(N,6,2) # [replica, state, action]
    elif algo == 'BayesQS':
        # Initialize Normal-gamma distribution parameters
        # NG_params_(batch, states, actions, {mu0,lambda,alpha,beta})
        NG_params = 1.5*np.ones([N,6,2,4])
    elif algo == 'Infomax':
        NG_params = 1.5*np.ones([N,6,2,4])
    
    # Simulation
    t_start = time.time()
    t1 = time.time()
    for t in range(n):
        # Choose arm
        if algo == 'Qlearn':
            epsilon = 0.1
            
            # Optimal vs. random (1 = choose optimal)
            eps_factor = (np.random.random(N) > epsilon).astype(int)
            
            # Optimal actions
            action_opt = np.zeros(N,dtype=int)
            for i in range(N):
                action_opt[i] = int(np.argmax(Q[i][state[i]]))

            # Random actions
            action_rand = np.random.randint(0,2,N,dtype=int)
            
            # Actual actions
            action = eps_factor * action_opt + (1-eps_factor) * action_rand

        elif algo == 'BayesQS':
            action = np.zeros(N, dtype = int)
            
            # Loop through batch
            for i in range(N):
                # Sample mu from normal-gamma distribution for each action
                mu_0,_ = sample_NG(NG_params[i][state[i]][0])
                mu_1,_ = sample_NG(NG_params[i][state[i]][1])
                
                # Choose the action corresponding to the optimal draw
                action[i] = int(mu_1 > mu_0)
                
        elif algo == 'Infomax':
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

        # Play
        chance = np.random.random(N) # Random number between 0 and 1; slip prob.
        
        # 1 = return to s0
        return_to_s0 = ((action==0)&(chance>0.2))|((action==1)&(chance<=0.2)).astype(int)
        
        # Proposed new state (possibly over 5)
        state_new = ((state+1)*(1-return_to_s0)).astype(int)
        
        # Reward
        r = return_to_s0*2 + (state_new > 5)*10
        
        # New state
        state_new = np.minimum(state_new, 5)
        
        # Keep track of cumulative rewards
        curr_cum_reward += r
        # Record cumulative rewards
        if t % n_rec == 0:
            cum_reward[:,int(t/n_rec)] = curr_cum_reward
            plays[int(t/n_rec)] = t+1
            
        # Update parameters
        if algo == 'Qlearn':
            alpha = 0.01 # Learning rate
            
            for i in range(N):
                Q[i,state[i],action[i]] = (1-alpha)*Q[i,state[i],action[i]] + alpha * (r[i] + gamma*max(Q[i,state_new[i]]))
                
        elif algo == 'BayesQS' or algo == 'Infomax': # Same posterior
            # Loop through batch
            for i in range(N):
                # Current state
                s_s = state[i]
                # Current action performed
                a_s = action[i]
                # Next state
                s_t = state_new[i]
                # Optimal action in next state t
                a_t = np.argmax(NG_params[i][s_t][:,0])
            
                # E[R_t] and E[R_t^2]
                ER = NG_params[i][s_t][a_t][0]
                ER2 = (NG_params[i][s_t][a_t][1] + 1)/NG_params[i][s_t][a_t][1] * NG_params[i][s_t][a_t][3]/(NG_params[i][s_t][a_t][2]-1) + NG_params[i][s_t][a_t][0]**2
            
                # Moments
                M1 = r[i] + gamma*ER
                M2 = r[i]**2 + 2*gamma*r[i]*ER + gamma**2*ER2
                
                # Update parameters
                mu0,lamb,alpha,beta = NG_params[i][s_s][a_s]
    
                NG_params[i][s_s][a_s][0] = (lamb*mu0 + M1)/(lamb + 1)
                NG_params[i][s_s][a_s][1] += 1
                NG_params[i][s_s][a_s][2] += 0.5
                NG_params[i][s_s][a_s][3] += 0.5*(M2-M1**2) + (lamb*(M1-mu0)**2)/(2*(lamb+1))
            
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
    
    output = {'cum_reward':cum_reward, 'plays':plays}
    
    if algo == 'Qlearn':
        output['Q'] = Q
    
    return output
    
# Plotting functions ----------------------------------------------------------------------------

# Plot reward
def plot_reward(plays, cum_reward, name, savefig):
    plt.figure()
    # Optimal reward
    #cum_reward_opt = p1 * plays - p1 # First optimal action is just transitioning
    #plt.plot(plays, cum_reward_opt, 'k', linewidth=3, label='Optimal')
    plt.plot(plays, cum_reward, 'r', linewidth=3, label='Q-learning')
    plt.legend(loc='best')
    #plt.gca().set_xscale('log',basex=10)
    #plt.ylim([0,max(cum_reward_opt)*1.5])
    plt.xlabel('Total number of plays')
    plt.ylabel('Reward')
    plt.title('Cumulative reward')
    if savefig:
        plt.savefig('Chain_reward_' + name + '.png')
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    # Parse inputs ------------------------------------------------------------------------------
    # Example command: python BayesQ.py --N 10 --algo BayesQS --verbose 1
    parser = argparse.ArgumentParser(description='2-armed bandit')
    parser.add_argument("--gamma", default=0.99, type=int, help="Discount factor")
    parser.add_argument("--n", default=1000, type=int, help="Total number of turns")
    parser.add_argument("--N", default=1, type=int, help="Replicas (For averaging)")
    parser.add_argument("--n_rec", default=1, type=int, help="Record every n turns")
    parser.add_argument("--algo", default='Qlearn', type=str, help="Decision algorithm; 'Qlearn', 'BayesQS'")
    parser.add_argument("--seed", default=111, type=int, help="Random seed")
    parser.add_argument("--savefig", default=0, type=int, help="Save figures")
    parser.add_argument("--verbose", default=0, type=int, help="Print messages")

    args = parser.parse_args()

    # Run main function -------------------------------------------------------------------------
    output = main(args)
    
    # Save output -------------------------------------------------------------------------------

    name =  'Chain' + \
            '_n=' + str(args.n) + \
            '_N=' + str(args.N) + \
            '_n_rec=' + str(args.n_rec) + \
            '_algo=' + str(args.algo) + \
            '_seed=' + str(args.seed)
                
    #pickle.dump(output, open(name + '.pickle',"wb"))
    
    # Extract outputs ---------------------------------------------------------------------------
    cum_reward = output['cum_reward']
    if 'Q' in output:
        Q = output['Q']
    plays = output['plays']
    
    cum_reward_mean = np.mean(cum_reward,axis=0)

    # Plot cumulative reward --------------------------------------------------------------------
    plot_reward(plays,cum_reward_mean,name,args.savefig)