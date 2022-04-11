import gym
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable

# Load environment
env = gym.make('FrozenLake-v0')

# Define the neural network mapping 16x1 one hot vector to a vector of 4 Q values
# and training loss
# TODO: define network, loss and optimiser(use learning rate of 0.1).

def get_one_hot(x):
    v = torch.zeros(16)
    v[x] = 1
    return v

network = nn.Sequential(nn.Linear(16, 4,bias=False))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(network.parameters(), lr=0.01)

# Implement Q-Network learning algorithm

# Set learning parameters
y = .99
e = 0.1
num_episodes = 2000
# create lists to contain total rewards and steps per episode
jList = []
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    # The Q-Network
    while j < 99:
        j += 1
        # 1. Choose an action greedily from the Q-network
        #    (run the network for current state and choose the action with the maxQ)
        # TODO: Implement Step 1
        Q = network(get_one_hot(s))
        a = torch.argmax(Q).item()
        
        # 2. A chance of e to perform random action
        if np.random.rand(1) < e:
          a = env.action_space.sample()
        
        # 3. Get new state(mark as s1) and reward(mark as r) from environment
        s1, r, d, _ = env.step(a)

        # 4. Obtain the Q'(mark as Q1) values by feeding the new state through our network
        # TODO: Implement Step 4
        Q1 = network(get_one_hot(s1))

        # 5. Obtain maxQ' and set our target value for chosen action using the bellman equation.
        # TODO: Implement Step 5
        Qt = Variable(Q.data)
        Qt[a] = r + torch.mul(y, torch.max(Q1).item())
        Q = network(get_one_hot(s))

        # 6. Train the network using target and predicted Q values (model.zero(), forward, backward, optim.step)
        # TODO: Implement Step 6
        network.zero_grad()
        loss = criterion(Qt, Q)
        loss.backward()
        optimizer.step()
        
        rAll += r
        s = s1
        if d == True:
            #Reduce chance of random action as we train the model.
            e = 1./((i/50) + 10)
            break
    jList.append(j)
    rList.append(rAll)

# Reports
print("Score over time: " + str(sum(rList)/num_episodes))
