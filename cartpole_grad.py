from email import policy
import numpy as np
import gym
import algo
import random
import NN
import torch
from torch.distributions import Categorical

def initialize_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        #print(f'in here with {classname}')
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

def main():

    # Make cartpole environment
    env = gym.make('CartPole-v1')
    
    '''
    env.observation_space.low[1] = -4
    env.observation_space.low[3] = -4
    env.observation_space.high[1] = 4
    env.observation_space.high[3] = 4
    '''


    len_state_space = len(env.observation_space.high)
    policygrad_params = {'gamma':0.99,
                    'alpha':0.1,
                    'epsilon': 1,
                    'len_state': len_state_space,
                    'len_action': env.action_space.n}

    #create policy network object
    PolNet = NN.PolicyNet(policygrad_params['len_state'], policygrad_params['len_action'])
    PolNet.apply(initialize_weights)
    loss = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(PolNet.parameters(), lr=0.001)

    #Start training
    for i in range(6000):
        done = False
        curr_state = env.reset()
        traj_dict = {'state': [],
                    'reward': [],
                    'action': []}
        traj_len =0
        PolNet.eval()
        while not done:
            #rollout policy using network inference
            #print(curr_state)
            traj_dict['state'].append(curr_state)
            with torch.no_grad():
                curr_action_dist = PolNet(torch.Tensor(curr_state))
            #print(curr_action_dist)
            #print(curr_state)
            action_dist = Categorical(curr_action_dist)
            curr_action = action_dist.sample()
            #print(curr_action)
            curr_state,r, done, _= env.step(np.array(curr_action))
            traj_dict['action'].append(curr_action)
            traj_dict['reward'].append(r)
            traj_len += 1

        traj_dict['dis_r'] = np.zeros(traj_len)
        # Find accumulated discounted reward
        for i in reversed(range(traj_len)):
            if i == traj_len - 1:
                traj_dict['dis_r'][i] = traj_dict['reward'][i]
            else:
                traj_dict['dis_r'][i] = traj_dict['reward'][i] + policygrad_params['gamma'] * traj_dict['dis_r'][i+1]

        opt.zero_grad()
        PolNet.train()

        #Start Training
        output_tot = 0
        for i in range(traj_len):
            s = traj_dict['state'][i]
            dis_r = traj_dict['dis_r'][i]
            a_label = traj_dict['action'][i]

            #inference
            act = PolNet(torch.Tensor(s))
            '''
            action_dist = Categorical(act)
            log_probs = -action_dist.log_prob(a_label) 
            '''
            log_probs = -loss(act, a_label)
            output = log_probs * dis_r

            output_tot += output

        output_tot.backward()
        opt.step()
    
        rew_array = np.sum(np.array(traj_dict['reward']))

        print(f'Episode collected and trained with accumulated reward of {rew_array}')



if __name__== "__main__":

    main()