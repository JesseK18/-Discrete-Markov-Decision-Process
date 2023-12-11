#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Programming 
Practical for course 'Symbolic AI'
2020, Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from world import World

class Dynamic_Programming:

    def __init__(self):
        self.V_s = None # will store a potential value solution table
        self.Q_sa = None # will store a potential action-value solution table
        
    def value_iteration(self,env,gamma = 1.0, theta=0.001):
        ''' Executes value iteration on env. 
        gamma is the discount factor of the MDP
        theta is the acceptance threshold for convergence '''
        
        print("Starting Value Iteration (VI)")
         
        # initialize value table
        V_s = np.zeros(env.n_states)
        self.V_s = np.zeros(env.n_states)

        # Initialize max difference (delta)
        max_diff = 1

        #Check for convergence

        while max_diff > theta:
            max_diff = 0
            for s in env.states:
                max_val = 0
                x = V_s[s]

                for a in env.actions:
                    # Compute state value
                    s_next, reward = env.transition_function(s, a)
                    reward += gamma * V_s[s_next]
                    max_val = max(max_val, reward)

                V_s[s] = max_val
                max_diff = max(max_diff, abs(x - V_s[s]))

            self.V_s = V_s
        print(self.V_s)
        return

    def Q_value_iteration(self,env,gamma = 1.0, theta=0.001):
        ''' Executes Q-value iteration on env. 
        gamma is the discount factor of the MDP
        theta is the acceptance threshold for convergence '''

        print("Starting Q-value Iteration (QI)")
        # initialize state-action value table
        Q_sa = np.zeros([env.n_states,env.n_actions])

        #Initialize action mapping to map integer actions to string actions so we can just call 'a' further in this function
        action_mapping = {
            0: "up",
            1: "down",
            2: "left",
            3: "right",
        }

        # Run loop until convergence (delta is smaller then theta)
        while True:
            max_diff = 0
            for s in range(env.n_states):
                max_val = 0
                for a in range(env.n_actions):
                    next_state, reward = env.transition_function(s, action_mapping[a])
                    q = reward + gamma * Q_sa[next_state][a]
                    max_val = max(max_val, q)
                # Update max difference with maximum value between old and new Q-values
                max_diff = np.maximum(max_diff, np.abs(Q_sa[s] - max_val))
                Q_sa[s] = max_val
            # Check for convergence
            if max_diff.all() < theta:
                break
            self.Q_sa = Q_sa

        return
                
    def execute_policy(self,env,table='V'):
        ## Execute the greedy action, starting from the initial state
        env.reset_agent()
        print("Start executing. Current map:") 
        env.print_map()
        while not env.terminal:
            current_state = env.get_current_state() # this is the current state of the environment, from which you will act
            available_actions = env.actions
            # Compute action values
            if table == 'V' and self.V_s is not None:
                for a in available_actions:
                    s_next, reward = env.transition_function(current_state, a)
                    if (self.V_s[current_state] == self.V_s[s_next] + reward):
                        greedy_action = a
                        break

                
            
            elif table == 'Q' and self.Q_sa is not None:
                #Action mapping as well, but then reversed in comparison to Q_value_iteration
                action_mapping = {
                    "up": 0,
                    "down": 1,
                    "left": 2,
                    "right": 3,
                }
                #This works the same as if table == 'V', but then the part 'action_mapping[a]' is added so it works for Q_sa
                for a in available_actions:
                    next_state, reward = env.transition_function(current_state,a)
                    if(self.Q_sa[current_state][action_mapping[a]] == self.Q_sa[next_state][action_mapping[a]] + reward):
                        greedy_action = a
                        break
                
                
            else:
                print("No optimal value table was detected. Only manual execution possible.")
                greedy_action = None


            # ask the user what he/she wants
            while True:
                if greedy_action is not None:
                    print('Greedy action= {}'.format(greedy_action))    
                    your_choice = input('Choose an action by typing it in full, then hit enter. Just hit enter to execute the greedy action:')
                else:
                    your_choice = input('Choose an action by typing it in full, then hit enter. Available are {}'.format(env.actions))
                    
                if your_choice == "" and greedy_action is not None:
                    executed_action = greedy_action
                    env.act(executed_action)
                    break
                else:
                    try:
                        executed_action = your_choice
                        env.act(executed_action)
                        break
                    except:
                        print('{} is not a valid action. Available actions are {}. Try again'.format(your_choice,env.actions))
            print("Executed action: {}".format(executed_action))
            print("--------------------------------------\nNew map:")
            env.print_map()
        print("Found the goal! Exiting \n ...................................................................... ")
    

def get_greedy_index(action_values):
    ''' Own variant of np.argmax, since np.argmax only returns the first occurence of the max. 
    Optional to uses '''
    return np.where(action_values == np.max(action_values))
    
if __name__ == '__main__':
    env = World('prison.txt') 
    DP = Dynamic_Programming()

    # Run value iteration
    input('Press enter to run value iteration')
    optimal_V_s = DP.value_iteration(env)
    input('Press enter to start execution of optimal policy according to V')
    DP.execute_policy(env, table='V') # execute the optimal policy
    
    # Once again with Q-values:
    input('Press enter to run Q-value iteration')
    optimal_Q_sa = DP.Q_value_iteration(env)
    input('Press enter to start execution of optimal policy according to Q')
    DP.execute_policy(env, table='Q') # execute the optimal policy

