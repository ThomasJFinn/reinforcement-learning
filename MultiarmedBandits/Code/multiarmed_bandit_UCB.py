import numpy as np
import random

def multiarmed_bandit_UCB(num_bandits, q_mean, steps, weight):
    """
    Performs multi-armed bandit algorithm using upper-confidence-bound action selection
    to find the optimal bandit arm

    Parameters:
        num_bandits (int): The number of bandit arms
        q_mean (array-like): The true mean value for each bandit arm
        steps (int): The number of steps to run the algorithm
        weight (float): Weight paramter in upper-confidence-bound action selection

    Returns:
        avg_reward (numpy array): The average reward at each step
        opt_ratio (numpy array): The ratio of optimal actions at each step
    """
    
    avg_reward = np.zeros(steps + 1)
    opt_ratio = np.zeros(steps + 1)
    opt_bandit = np.argmax(q_mean)
    
    # Initialise sample-averages and action count
    Q = np.zeros(num_bandits)
    action_count = np.zeros(num_bandits)
    
    for i in range(1, steps + 1):
        # Perform upper-confidence-bound action selection
        Q2 = Q + weight * np.sqrt(np.log(i) / (action_count + 1e-6))
        action = np.argmax(Q2)
            
        # Incremental implementation for updating action values
        reward = np.random.normal(q_mean[action], 1)
        action_count[action] += 1
        Q[action] += (1 / action_count[action]) * (reward - Q[action])
    
        # Update average reward and optimal action ratio
        avg_reward[i] += (1 / i) * (((i - 1) * avg_reward[i - 1]) + reward)
        opt_ratio[i] += (1 / i) * (((i - 1) * opt_ratio[i - 1]) + (action == opt_bandit))
    
    avg_reward = avg_reward[1:]  # Remove the initial element 
    opt_ratio = opt_ratio[1:]  # Remove the initial element 
         
    return avg_reward, opt_ratio