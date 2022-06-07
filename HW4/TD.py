from environment import Easy21
import numpy as np

def run_exp(episodes = int(1e4), alpha_par = None, gamma_par = None):
    env = Easy21()
    actions = [0, 1]

    def reset():
        Q = np.zeros((23, 12)) # state-action value
        NSA = np.zeros((23, 12, len(actions))) # state-action counter
        wins = 0

        return Q, NSA, wins

    V, NSA, wins = reset()

    # number of times state s has been visited
    NS = lambda p, d: np.sum(NSA[p, d])

    # step size and gamma

    def gamma(p,d,a):
        if gamma_par is None:
            return 1/(1+NS(p, d)*10)
        else:
            return gamma_par
    def alpha(p,d,a):
        if alpha_par is None:
            return 1/(1+NSA[p, d, a])
        else:
            return alpha_par

    def player_td0(p, d):
        return 0 if p < 18 else 1

    lmds = [0.0]


    for li, lmd in enumerate(lmds):

        V, NSA, wins = reset()

        for episode in range(episodes):
            env.init_deck()

            terminated = False
            p, d = env.initGame()

            SA = list()
            # # first action is draw anyway - starts with 1 card in hand
            a = player_td0(p, d)

            # Sample Environment
            while not terminated:
                pPrime, dPrime, r, terminated = env.step(p, d, a)

                if not terminated:
                    # if callable(gamma(p,d,a)) and gamma(p,d,a).__name__ == "<lambda>":
                    #     print("error")
                    aPrime = player_td0(pPrime, dPrime)
                    tdError = r + gamma(p,d,a) * V[pPrime, dPrime] - V[p, d] # used an adaptive gamma here
                else:
                    tdError = r - V[p, d]

                NSA[p, d, a] += 1
                SA.append([p, d, a])

                for (_p, _d, _a) in SA:
                    V[_p, _d] += alpha(_p, _d, _a) * tdError

                if not terminated:
                    p, d, a = pPrime, dPrime, aPrime

            # bookkeeping
            if r == 1:
                wins += 1

    return wins

GAMMA = 0.65
ALPHA = 0.78
episodes = 100000
best_alpha, best_gamma, best_prob = 0.0,0.0,0.0
mean,mean_ada = [],[]

wins = run_exp(episodes) # run with adaptive alpha gamma
print(f"TD(0) Win Probability is {float(wins)/episodes:.4f} Over {episodes} runs")

# for a in np.linspace(0.01,0.99, 1):
#     for g in np.linspace(0.01,0.99,1):
#         for i in range(5):
#             wins = run_exp(episodes) # run with adaptive alpha gamma
#             # print(f"TD(0) Probability for win is {float(wins)/episodes:.4f} Adaptive Gamma and Alpha")
#
#             # wins_const = run_exp(episodes, a,g) # run with set gamma and alpha
#             # print(f"TD(0) Probability for win is {float(wins)/episodes:.4f} Gamma = {GAMMA} and Alpha = {ALPHA}")
#
#             if float(wins)/episodes > best_prob:
#                 best_prob = float(wins)/episodes
#
#             # mean.append(float(wins_const)/episodes)
#             mean_ada.append(float(wins)/episodes)


# print(f"TD(0) Probability const Average is {np.mean(mean):.4f}, adaptive is {np.mean(mean_ada):.4f}")



