from environment import Easy21
import numpy as np

def run_exp(episodes = int(1e4), alpha_par = None, gamma_par = None, threshold = 0.95, policy = 'softmax', print_flag=True):
    env = Easy21()
    N0 = 4
    actions = [0, 1]
    optimal_action_list = [12*[None] for _ in range(23)]

    def reset():
        Q = np.zeros((23, 12, len(actions))) # state-action value
        NSA = np.zeros((23, 12, len(actions))) # state-action counter
        wins = 0
        wins_state_arr = np.zeros((23, 12, 1)) # state-action value


        return Q, NSA, wins, wins_state_arr

    Q, NSA, wins, wins_per_state = reset()

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
            return 1/(1+NSA[p, d, a]*5)
        else:
            return alpha_par

    # exploration probability
    epsilon = lambda p, d: N0 / (N0 + NS(p, d))

    def epsilonGreedy(p, d, episode):
        if np.random.random() < epsilon(p, d):
            # explore
            action = np.random.choice(actions)

        else:
            # exploit
            action = np.argmax( [Q[p, d, a] for a in actions] )

        return action

    def softmax(p, d, episode):
        if episode < episodes * threshold:
            # explore
            beta = 1.0 / 0.5**(int(episode/25000))
            denom = np.sum([np.exp(beta *Q[p, d, a]) for a in actions])
            action = np.argmax( [np.exp(beta *Q[p, d, a])/denom for a in actions])

        else:
            # exploit
            action = np.argmax( [Q[p, d, a] for a in actions] )

        return action

    policy = softmax if policy == 'softmax' else epsilonGreedy
    lmds = [0.0]

    optimal_action_arr = np.zeros((23, 12, 1))  # state-action value

    for li, lmd in enumerate(lmds):

        Q, NSA, wins, _ = reset()

        for episode in range(episodes):

            env.init_deck()

            terminated = False
            p, d = env.initGame()
            # inital state and first action
            a = policy(p, d, episode)
            SA = list()

            # Sample Environment
            while not terminated:

                pPrime, dPrime, r, terminated = env.step(p, d, a)

                if not terminated:
                    aPrime = policy(pPrime, dPrime, episode)
                    SARSAError = r + gamma(p,d,a) * Q[pPrime, dPrime, aPrime] - Q[p, d, a]
                else:
                    SARSAError = r - Q[p, d, a]

                NSA[p, d, a] += 1
                SA.append([p, d, a])

                for (_p, _d, _a) in SA:
                    Q[_p, _d, _a] += alpha(_p, _d, _a) * SARSAError

                if not terminated:
                    p, d, a = pPrime, dPrime, aPrime

            # bookkeeping
            if r == 1:
                wins += 1
                for (_p, _d, _a) in SA:
                    wins_per_state[_p, _d, 0] += 1
    for p in range(23):
        for d in range(12):
            optimal_action_arr[p,d,0] = np.argmax( [Q[p, d, a] for a in actions] )
            optimal_action_list[p][d] = 'stand' if optimal_action_arr[p,d,0] == 1 else 'hit'
    for p in range(23):
        for d in range(12):
            try:
                wins_per_state[p, d, 0] = wins_per_state[p, d, 0] / NS(p, d)
                if not np.isnan(wins_per_state[p, d, 0]) and print_flag:
                    print(f'State [{p},{d}] Prob. to win is {wins_per_state[p, d, 0]:.2f}, optimal action is {optimal_action_list[p][d]}')
            except:
                continue


    return wins, wins_per_state , optimal_action_list

GAMMA = 0.65
ALPHA = 0.78
episodes = 100000
THRESHOLD = 0.83
POLICY = 'softmax'
best_alpha, best_gamma, best_prob , best_thresh= 0.0,0.0,0.0,0.0
best_prob_arr = None
mean,mean_ada,mean_eps = [],[],[]
wins, state_win_prob_arr, optimal_action_arr = run_exp(episodes, threshold=THRESHOLD, policy='epsilon',
                                                       print_flag=True)  # run with adaptive alpha gamma
print(f"SARSA Probability for win is {float(wins) / episodes:.4f} Adaptive Gamma and Alpha, Policy = epsilonGreedy")
wins, state_win_prob_arr, optimal_action_arr = run_exp(episodes, threshold=THRESHOLD, policy=POLICY,
                                                       print_flag=True)  # run with adaptive alpha gamma
print(f"SARSA Probability for win is {float(wins) / episodes:.4f} Adaptive Gamma and Alpha, Policy = {POLICY}")

# for i in range(5):
#     wins, state_win_prob_arr, optimal_action_arr = run_exp(episodes, threshold=THRESHOLD, policy='epsilon', print_flag=False)  # run with adaptive alpha gamma
#     print(f"SARSA Probability for win is {float(wins)/episodes:.4f} Adaptive Gamma and Alpha, Policy = epsilonGreedy")
#     print()
#     mean_eps.append(float(wins)/episodes)
#
#     wins, state_win_prob_arr, optimal_action_arr = run_exp(episodes, threshold=THRESHOLD, policy=POLICY, print_flag=False)  # run with adaptive alpha gamma
#     print(f"SARSA Probability for win is {float(wins)/episodes:.4f} Adaptive Gamma and Alpha, Policy = {POLICY}")
#     mean_ada.append(float(wins)/episodes)


# print(f'{np.mean(mean_ada):.4f} - SoftMax, {np.mean(mean_eps):.4f} - EpsilonGreedy')
# for thr in np.linspace(0.01,0.9, 10):
#     for a in np.linspace(0.01,0.99,1):
#         for g in np.linspace(0.01, 0.99, 1):
#             for i in range(5):
#                 wins, state_win_prob_arr, optimal_action_arr = run_exp(episodes, threshold=thr, print_flag=False) # run with adaptive alpha gamma
#                 # print(f"TD(0) Probability for win is {float(wins)/episodes:.4f} Adaptive Gamma and Alpha")
#
#                 # wins_const, state_win_prob_arr_const, optimal_action_arr_const = run_exp(episodes, ALPHA,GAMMA,  threshold=thr) # run with set gamma and alpha
#                 # print(f"TD(0) Probability for win is {float(wins)/episodes:.4f} Gamma = {GAMMA} and Alpha = {ALPHA}")
#
#                 if float(wins)/episodes > best_prob:
#                     best_prob = float(wins)/episodes
#                     # best_prob_arr = state_win_prob_arr_const
#                     best_alpha, best_gamma = a, g
#                     best_thresh = thr
#
#                 # mean.append(float(wins_const)/episodes)
#                 mean_ada.append(float(wins)/episodes)
#
#
# print(f"SARSA Probability const Average is {np.mean(mean):.4f}, adaptive is {np.mean(mean_ada):.4f}, best threshold = {best_thresh:.2f} best Alpha = {best_alpha:.2f}, BestGamma = {best_gamma:.2f}")
# # print(best_prob_arr)