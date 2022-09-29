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
    
    policy = algo.DQN(height, length)

















if __name__ == "__main__":
    
    main()
    