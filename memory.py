import random
import numpy as np

class replay_memory():
    def __init__(self,state_size,discrete_actions_size,continuou_action_size):
        self.state_size=state_size
        self.discrete_action_size=discrete_actions_size
        self.continuou_action_size=continuou_action_size
        self.memory_size = 5000
        self.discrete_action = np.empty((self.memory_size, self.discrete_action_size), dtype=np.int32)
        self.continuou_action = np.empty((self.memory_size, self.continuou_action_size), dtype=np.float64)
        self.rewards = np.empty(self.memory_size, dtype=np.float64)
        self.prestate = np.empty((self.memory_size, self.state_size), dtype=np.float64)
        self.poststate = np.empty((self.memory_size, self.state_size), dtype=np.float64)
        self.batch_size = 300          #每一轮训练批次
        self.count = 0
        self.current = 0

    def add(self, prestate, poststate, reward,continuou_action, discrete_action):
        self.discrete_action[self.current] = discrete_action
        self.continuou_action[self.current] = continuou_action
        self.rewards[self.current] = reward
        self.prestate[self.current] = prestate
        self.poststate[self.current] = poststate
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.memory_size

    def sample(self):
        if self.count < self.batch_size:
            indexes = range(0, self.count)
        else:
            indexes = random.sample(range(0,self.count), self.batch_size)
        prestate = self.prestate[indexes]
        poststate = self.poststate[indexes]
        discrete_action = self.discrete_action[indexes]
        continuou_action = self.continuou_action[indexes]
        rewards = self.rewards[indexes]
        return prestate, poststate,discrete_action,continuou_action, rewards
