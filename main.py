import numpy as np
import env
import random

def init_Q(h, l):
    Q = np.zeros((h, l, 4))
    return Q

def epsilon_greedy(epsilon, Q, curr_state):
    dice = random.uniform(0,1)
    if dice < epsilon:
        act = random.randint(0,3)
        #print(act)
    else:
        print(curr_state)
        q_curr_state = Q[curr_state[0], curr_state[1],:]
        #print(q_curr_state.shape)
        act = np.argmax(q_curr_state)

    return act



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
    alpha = 0.1
    cliff = env.cliff_env(length,height)
    print(cliff.curr_state)
    ## Implement SARSA
    Q = init_Q(height, length)
    #print(Q)
    epsilon = 0.01
    for i in range(100000):
        curr_state = cliff.curr_state
        curr_act = epsilon_greedy(epsilon, Q, curr_state)
        
        act_word = key[curr_act]
        cliff.action(act_word)
        cliff.check_endstate()
        curr_reward = cliff.curr_reward
        next_state = cliff.curr_state
        next_act = epsilon_greedy(epsilon, Q, next_state)

        # Update Q values
        td_error = curr_reward + gamma* Q[next_state[0], next_state[1],next_act] - Q[curr_state[0], curr_state[1],curr_act]
        update = Q[curr_state[0], curr_state[1],curr_act] + alpha*td_error
        Q[curr_state[0], curr_state[1],curr_act] = update

        #check if episode ended
        chk_end = cliff.end
        if chk_end == 1:
            print(f'episode ended with {cliff.acc_reward} accumulated reward')
            cliff.reset()
    print(Q)
    print(Q[0,0,:])
    print(Q[-1,-1,:])
    print(cliff.visited_states)











if __name__ == "__main__":
    main()