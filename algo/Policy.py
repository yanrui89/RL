import numpy as np
import random
import copy

class sarsa:
    def __init__(self, h, l):
        self.Q = self.init_Q(h,l)
        self.prevQ = self.init_Q(h,l) + 1
        self.converge = 0
        self.max_diff_Q = 0
        


    def init_Q(self,h, l):
        Q = np.zeros((h, l, 4))
        return Q
    
    
    def epsilon_greedy(self,epsilon, curr_state):
        dice = random.uniform(0,1)
        if dice < epsilon:
            act = random.randint(0,3)
            #print(act)
        else:
            print(curr_state)
            q_curr_state = self.Q[curr_state[0], curr_state[1],:]
            #print(q_curr_state.shape)
            act = np.argmax(q_curr_state)

        return act

    def update_sarsa(self, curr_reward, gamma, next_state, next_act, curr_state, curr_act, alpha):
        td_error = curr_reward + gamma* self.Q[next_state[0], next_state[1],next_act] - self.Q[curr_state[0], curr_state[1],curr_act]
        update = self.Q[curr_state[0], curr_state[1],curr_act] + alpha*td_error
        self.Q[curr_state[0], curr_state[1],curr_act] = update

    def find_best_path(self):
        best_path = np.argmax(self.Q, axis = 2)

        return best_path

    def chk_converge(self):
        diff_Q = np.abs(self.Q - self.prevQ)
        self.max_diff_Q = np.max(diff_Q)
        print(self.max_diff_Q)
        self.prevQ = copy.deepcopy(self.Q)
        if self.max_diff_Q < 1e-15:
            self.converge = 1
        
