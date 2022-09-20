import numpy as np
import env
import random
import algo



def main():
    key = {
        0 : 'left',
        1 : 'up',
        2 : 'right',
        3 : 'down'
    }
    height = 5
    length = 10
    gamma = 0.99
    alpha = 0.2
    cliff = env.cliff_env(length,height)
    policy = algo.sarsa(height, length)
    #print(cliff.curr_state)
    ## Implement SARSA
    epsilon = 0.2
    while policy.converge == 0:
        curr_state = cliff.curr_state
        curr_act = policy.epsilon_greedy(epsilon, curr_state)
        act_word = key[curr_act]
        cliff.action(act_word)
        cliff.check_endstate()
        curr_reward = cliff.curr_reward
        next_state = cliff.curr_state
        next_act = policy.epsilon_greedy(epsilon, next_state)

        # Update Q values
        policy.update_sarsa(curr_reward, gamma, next_state, next_act, curr_state, curr_act, alpha)

        #check if episode ended
        chk_end = cliff.end
        if chk_end == 1:
            print(f'episode ended with {cliff.acc_reward} accumulated reward')
            cliff.reset()

        policy.chk_converge()


    #print(policy.Q)
    #print(policy.Q[0,0,:])
    #print(policy.Q[-1,-1,:])
    #print(cliff.visited_states)


    #Plot the actual path 
    bp = policy.find_best_path()
    print(bp)












if __name__ == "__main__":
    main()