import numpy as np
import random
import copy
import NN
import torch

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

class q_learning(sarsa):

    def __init__(self,h, l):
        super().__init__(h,l)

    def greedy(self, curr_state):

        print(curr_state)
        q_curr_state = self.Q[curr_state[0], curr_state[1],:]
        #print(q_curr_state.shape)
        act = np.argmax(q_curr_state)

        return act

    def update_qlearn(self, curr_reward, gamma, next_state, next_act, curr_state, curr_act, alpha):
        td_error = curr_reward + gamma* self.Q[next_state[0], next_state[1],next_act] - self.Q[curr_state[0], curr_state[1],curr_act]
        update = self.Q[curr_state[0], curr_state[1],curr_act] + alpha*td_error
        self.Q[curr_state[0], curr_state[1],curr_act] = update
        
        
class DQN:
    
    def __init__(self):

        self.q = NN.MLP(2,4)
        self.qtarget = NN.MLP(2,4)
        self.repmem = NN.memrep()
        self.optim = torch.optim.SGD(self.q.parameters(), lr = 0.01,momentum = 0.9)
        self.optim.zero_grad()
        
    def learnq(self,batch, gamma):
        
        self.optim.zero_grad()
        num_in_rep = self.repmem.chk_num_sample()
        min_batch = np.min(np.array([batch, num_in_rep]))
        
        #sample sarsa from rep buffer
        
        samples = self.repmem.draw_sample(min_batch)
        
        sample_torch = torch.tensor(np.array(samples))
        sample_torch.to('cuda')
        
        #find target
        y = sample_torch[:,3] + gamma*(self.greedy(np.array(sample_torch[:,4:6])))*sample_torch[:,-1]
        input = self.q(torch.FloatTensor(sample_torch[:,0:2]))[:,sample_torch[:,2]]
        loss = torch.nn.MSELoss()
        output = loss(input,y)
        output.backward()
        self.optim.step()
        
        return
        
    def epsilon_greedy(self,epsilon, curr_state):
        dice = random.uniform(0,1)
        if dice < epsilon:
            act = random.randint(0,3)
            #print(act)
        else:
            #print(curr_state)
            with torch.no_grad():
                q_curr_state = self.q(torch.FloatTensor(np.array([curr_state])))
            #print(q_curr_state.shape)
                act = np.argmax(q_curr_state)
        return act
    
    def greedy(self, curr_state):
        #print(curr_state)
        with torch.no_grad():
            q_curr_state = self.qtarget(torch.FloatTensor(curr_state))
            #print(q_curr_state.shape)
            act_value = np.max(q_curr_state)
        return act_value
    
    
    def upload_mem(self, s,a,r,s1,es):
        self.repmem.push(s[0],s[1],a,r,s1[0],s1[1], es)
        
        
    def find_best_path(self, starting_state):
        best_path = np.argmax(self.Q, axis = 2)

        return best_path
        
        
        
    
