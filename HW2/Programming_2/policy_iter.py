##################################
# Create env
import gym
env = gym.make('FrozenLake-v0')
env = env.env
print(env.__doc__)
print("")

#################################
# Some basic imports and setup
# Let's look at what a random episode looks like.

import numpy as np, numpy.random as nr, gym
import matplotlib.pyplot as plt
#%matplotlib inline
np.set_printoptions(precision=3)

# Seed RNGs so you get the same printouts as me
env.seed(0); from gym.spaces import prng; prng.seed(10)
# Generate the episode
env.reset()
for t in range(100):
    env.render()
    a = env.action_space.sample()
    ob, rew, done, _ = env.step(a)
    if done:
        break
assert done
env.render();

#################################
# Create MDP for our env
# We extract the relevant information from the gym Env into the MDP class below.
# The `env` object won't be used any further, we'll just use the `mdp` object.

class MDP(object):
    def __init__(self, P, nS, nA, desc=None):
        self.P = P # state transition and reward probabilities, explained below
        self.nS = nS # number of states
        self.nA = nA # number of actions
        self.desc = desc # 2D array specifying what each grid cell means (used for plotting)
mdp = MDP( {s : {a : [tup[:3] for tup in tups] for (a, tups) in a2d.items()} for (s, a2d) in env.P.items()}, env.nS, env.nA, env.desc)
GAMMA = 0.95 # we'll be using this same value in subsequent problems

print("")
print("mdp.P is a two-level dict where the first key is the state and the second key is the action.")
print("The 2D grid cells are associated with indices [0, 1, 2, ..., 15] from left to right and top to down, as in")
print(np.arange(16).reshape(4,4))
print("Action indices [0, 1, 2, 3] correspond to West, South, East and North.")
print("mdp.P[state][action] is a list of tuples (probability, nextstate, reward).\n")
print("For example, state 0 is the initial state, and the transition information for s=0, a=0 is \nP[0][0] =", mdp.P[0][0], "\n")
print("As another example, state 5 corresponds to a hole in the ice, in which all actions lead to the same state with probability 1 and reward 0.")
for i in range(4):
    print("P[5][%i] =" % i, mdp.P[5][i])
print("")

#################################
# Programing Question No. 2, part 1 - implement where required.

def compute_vpi(pi, mdp, gamma):
    # use pi[state] to access the action that's prescribed by this policy
    P = np.zeros((mdp.nS, mdp.nS))
    R = np.zeros(mdp.nS)

    for state in range(mdp.nS):
        R[state] = np.sum([tup[2] * tup[0] for tup in mdp.P[state][pi[state]]])  # compute R

        for tup in mdp.P[state][pi[state]]:
            P[state][tup[1]] += tup[0]  # compute P

    A = np.identity(mdp.nS) - gamma * P # Define A temp matrix for the solver

    # compute V
    V = np.linalg.solve(A, R)
    return V

actual_val = compute_vpi(np.arange(16) % mdp.nA, mdp, gamma=GAMMA)
print("Policy Value: ", actual_val)

#################################
# Programing Question No. 2, part 2 - implement where required.

def compute_qpi(vpi, mdp, gamma):
    Qpi = np.zeros([mdp.nS, mdp.nA]) # REPLACE THIS LINE WITH YOUR CODE
    for action in range(mdp.nA):
        for state in range(mdp.nS):
            sum_val = np.sum([tup[2] * tup[0] for tup in mdp.P[state][action]])
            states = [tup[1] for tup in mdp.P[state][action]]
            # sum for each state and action in Q_pi
            sum_val += gamma * np.sum([mdp.P[state][action][i][0] * vpi[st] for i, st in enumerate(states)])
            Qpi[state][action] = sum_val
    return Qpi

Qpi = compute_qpi(np.arange(mdp.nS), mdp, gamma=0.95)
print("Policy Action Value: ", actual_val)

#################################
# Programing Question No. 2, part 3 - implement where required.
# Policy iteration

def policy_iteration(mdp, gamma, nIt):
    Vs = []
    pis = []
    pi_prev = np.zeros(mdp.nS,dtype='int')
    pis.append(pi_prev)
    print("Iteration | # chg actions | V[0]")
    print("----------+---------------+---------")
    for it in range(nIt):
        # YOUR CODE HERE
        vpi = compute_vpi(pi_prev, mdp, gamma)
        qpi = compute_qpi(vpi, mdp, gamma)
        # you need to compute qpi which is the state-action values for current pi
        pi = qpi.argmax(axis=1)
        print("%4i      | %6i        | %6.5f"%(it, (pi != pi_prev).sum(), vpi[0]))
        Vs.append(vpi)
        pis.append(pi)
        pi_prev = pi

    for state in range(mdp.nS):
        iters = [i for i in range(1, nIt + 1)]
        vals = [Vs[i][state] for i in range(nIt)]
        plt.plot(iters, vals, label=f"state {state}")
        plt.xticks(iters)
        plt.legend(loc='upper right',
                   ncol=3, fancybox=True, shadow=True)
        plt.xlabel("Iteration")
        plt.ylabel("Value")
    return Vs, pis


Vs_PI, pis_PI = policy_iteration(mdp, gamma=0.95, nIt=20)
plt.plot(Vs_PI);
