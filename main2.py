import torch
import numpy as np
import env
import random
import algo
import parsers
import NN



def main():
    key = {
        0 : 'left',
        1 : 'up',
        2 : 'right',
        3 : 'down'
    }

    args = parsers.parse_args()
    
    height = args.height
    length = args.length
    gamma = args.gamma
    alpha = args.alpha
    epsilon = args.epsilon
    algo_name = args.algo_name

    cliff = env.cliff_env(length,height)
    
    policy = algo.DQN()
    
    for i in range(10000):
        print(i)
        curr_state = cliff.curr_state
        curr_act = policy.epsilon_greedy(epsilon, curr_state)
        act_word = key[curr_act]
        cliff.action(act_word)
        cliff.check_endstate()
        curr_reward = cliff.curr_reward
        next_state = cliff.curr_state
        #check if next state is end state
        end_state = cliff.check_endstate_2()
        
        #next_act = policy.greedy(next_state, end_state)
        
        #Upload tuple
        policy.upload_mem(curr_state,curr_act,curr_reward,next_state, end_state)
        policy.learnq(100,0.1)
        
        
        
        #check if episode ended
        chk_end = cliff.end
        if chk_end == 1:
            print(f'episode ended with {cliff.acc_reward} accumulated reward')
            cliff.reset()
            
    #check states
    cliff.reset()
    iter = 0
    while cliff.check_endstate != 0 and iter < 100:
        print('me here')
        curr_state = cliff.curr_state
        curr_act = policy.epsilon_greedy(epsilon, curr_state)
        act_word = key[curr_act]
        cliff.action(act_word)
        cliff.check_endstate()
        iter+= 1
        
    print(cliff.visited_states)
    
    

if __name__ == "__main__":
    
    main()
    