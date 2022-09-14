import numpy as np

class cliff_env:
    def __init__(self, length, height):
        self.length = length
        self.height = height
        self.start_state = np.array([0, 0])
        self.end_state = np.array([0, length-1])
        self.norm_reward = -1
        self.cliff_reward = -100
        self.reach_reward = 100
        self.curr_state = self.start_state
        self.end = 0
        self.acc_reward = 0
        self.curr_reward = 0
        self.visited_states = np.zeros((height, length))

    def check_endstate(self):
        rch_goal = (self.curr_state == self.end_state).all()
        if rch_goal == 1:
            self.acc_reward += self.reach_reward
            self.curr_reward = self.reach_reward
            self.end = 1
        else:
            if self.curr_state[0] == 0:
                if self.curr_state[1] != 0 or self.curr_state[1] != self.length-1 :
                    self.acc_reward += self.cliff_reward
                    self.curr_reward = self.cliff_reward
                    self.end = 1
            else:
                
                self.acc_reward += self.norm_reward
                self.curr_reward = self.norm_reward
                

    

    def action(self, act):
        x = self.curr_state[0]
        print(x)
        y = self.curr_state[1]
        if act == "left":
            if y != 0:
                self.curr_state = np.array([x, y-1])
        elif act == "right":
            if y != self.length -1:
                self.curr_state = np.array([x, y+1])
        elif act == "up":
            if x != self.height -1:
                print(self.height - 1)
                print('I am in')
                self.curr_state = np.array([x+1, y])
        elif act == "down":
            if x != 0:
                self.curr_state = np.array([x-1, y])

        self.visited_states[self.curr_state[0], self.curr_state[1]] += 1

    def reset(self):
        self.curr_state = self.start_state
        self.end = 0
        self.acc_reward = 0
        self.curr_reward = 0
        



