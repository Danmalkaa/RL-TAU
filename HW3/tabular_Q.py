from re import A, S
import gym
import numpy as np

# Load environment
env = gym.make('FrozenLake-v0')

# Implement Q-Table learning algorithm
#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
# Set learning parameters
lr = .8
y = .95
num_episodes = 2000
noise = 0.01
#create lists to contain total rewards and steps per episode
#jList = []
rList = []
Q_table_size = Q.shape[1]
for i in range(num_episodes):
    #Reset environment and get first new observation
    s = env.reset()
    rAll = 0 # Total reward during current episode
    d = False
    j = 0
    #The Q-Table learning algorithm
    while j < 99:
        j+=1
        # TODO: Implement Q-Learning
        # 1. Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s] + np.random.normal(0, noise, Q_table_size))
        # 2. Get new state and reward from environment
        new_s, reward, d, _ = env.step(a)
        # 3. Update Q-Table with new knowledge
        CAPITAL_GAMMA = reward + y * np.max(Q[new_s]) - Q[s][a]
        Q[s][a] = Q[s][a] + lr * CAPITAL_GAMMA
        # 4. Update total reward
        rAll += reward
        # 5. Update episode if we reached the Goal State
        if d:
          break
        s = new_s
    rList.append(rAll)

# Reports
print("Score over time: " +  str(sum(rList)/num_episodes))
print("Final Q-Table Values")
print(Q)
