import numpy as np
import env
import random
import algo
import parsers



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

    if algo_name == 'sarsa':
        policy = algo.sarsa(height, length)
    elif algo_name == 'qlearn':
        policy = algo.q_learning(height, length)

    while policy.converge == 0:
        curr_state = cliff.curr_state
        curr_act = policy.epsilon_greedy(epsilon, curr_state)
        act_word = key[curr_act]
        cliff.action(act_word)
        cliff.check_endstate()
        curr_reward = cliff.curr_reward
        next_state = cliff.curr_state

        if algo_name == 'sarsa':
            next_act = policy.epsilon_greedy(epsilon, next_state)
        elif algo_name == 'qlearn':
            next_act = policy.greedy(next_state)

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