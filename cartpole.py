import numpy as np
import gym
import algo
import random


def get_discrete_state(s, bins, len_state_space):
    new_s = []
    #print(len_state_space)
    for i in range(len_state_space):
        #print(i)
        new_s.append(np.digitize(s[i], bins[i]))
    return new_s


def create_bins(env, dis_Size, len_state_space):
    
    bins = []
    for i in range(len_state_space):
        item = np.linspace(
            env.observation_space.low[i] if (i == 0) or (i == 2) else -4,  # cap the max and min values of velocity and angular vel to be bbetween -4 and 4
            env.observation_space.high[i] if (i == 0) or (i == 2) else 4,
            num=dis_Size,
            endpoint=False)
        item = np.delete(item, 0)
        bins.append(item)
        print(bins[i])
    return bins

def choose_action(q_tables, state, params, mode):
    if mode == 'epsilon':
        dice = random.uniform(0,1)
    elif mode == 'greedy':
        dice = 2

    if dice < params['epsilon']:
        act = random.randint(0,params['len_action']-1)
        #print(act)
    else:
        q_curr_state = q_tables[tuple(state)]
        #print(q_curr_state.shape)
        act = np.argmax(q_curr_state)
    return act


def qlearn_update(curr_reward, curr_state, curr_act, next_state, next_act, q_tables, params):

    next_q = q_tables[tuple(next_state)]
    curr_q = q_tables[tuple(curr_state)]
    td_error = curr_reward + params['gamma']*next_q[next_act] - curr_q[curr_act]
    update = curr_q[curr_act] + params['alpha']*td_error

    full_tuple = tuple(curr_state) + (curr_act,)
    q_tables[full_tuple] = update

    return q_tables



def main():

    # Make cartpole environment
    env = gym.make('CartPole-v1')

    len_state_space = len(env.observation_space.high)
    qlearn_params = {'gamma':0.99,
                    'alpha':0.1,
                    'epsilon': 1,
                    'len_state': len_state_space,
                    'len_action': env.action_space.n}

    dis_Size = 20
    epsilon_decay = qlearn_params['epsilon'] / 5000


    len_state_space = len(env.observation_space.high)
    bins = create_bins(env, dis_Size, len_state_space)
    
    # create algo object
    #qlearn = algo.q_learning(dis_Size, dis_Size, env.action_space.n)

    # Initialize q_table for Cartpole, cartpole has 4 types of states, each state has 20 possible values. It has 1 type of action with 2 values
    q_tables = np.zeros((dis_Size, dis_Size, dis_Size, dis_Size, env.action_space.n))

    
    #Start training
    for i in range(6000):
        done = False
        curr_state = env.reset()
        accum_reward = 0
        while not done:
            curr_discrete_state = get_discrete_state(curr_state, bins, len_state_space)
            curr_act = choose_action(q_tables, curr_discrete_state, qlearn_params, 'epsilon')
            #print(curr_act)
            next_state,r, done, _= env.step(curr_act)

            next_discrete_state = get_discrete_state(next_state, bins, len_state_space)
            next_act = choose_action(q_tables, curr_discrete_state, qlearn_params, 'greedy')

            #update q table
            q_tables = qlearn_update(r, curr_discrete_state, curr_act, next_discrete_state, next_act, q_tables, qlearn_params)

            curr_state = next_state

            accum_reward += r
        
        print(f'Total reward for episdoe {i} is {accum_reward}')

        #if qlearn_params['epsilon'] <= 1 and i >= 3000:
        qlearn_params['epsilon'] -= epsilon_decay
    print('end')








     



# define function to convert to discrete state





if __name__== "__main__":

    main()