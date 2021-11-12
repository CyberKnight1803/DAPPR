import numpy as np 
import random 
from collections import deque

class ReplayBuffer:
    def __init__(self, size: int):
        self.buffer = deque(maxlen=size)
        self.maxSize = size 
        self.len = 0
    
    def sample(self, count):
        """
        Samples a random batch from replay memory buffer
        count: Batch Size
        """

        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)

        s = np.float32([arr[0] for arr in batch])
        a = np.float32([arr[1] for arr in batch])
        r = np.float32([arr[2] for arr in batch])
        s_ = np.float32([arr[3] for arr in batch])

        return s, a, r, s_ 

    def len(self):
        return self.len 
    
    def add(self, s, a, r, s_):
        """
        Add a transaction in memopry buffer
        s: current state 
        a: action taken
        r: reward recieved
        s_: next state 
        """

        transition = (s, a, r, s_)
        self.len += 1 

        if self.len > self.maxSize:
            self.len = self.maxSize 
        
        self.buffer.append(transition)
    