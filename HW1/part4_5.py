import gym
import numpy as np
import matplotlib.pyplot as plt
import pylab


def run_episode(env, weights):
    total_reward = 0
    observation = env.reset()
    for i in range(200): # Redundant because v0 stops after 200 rounds / v1 after 500 rounds
        # env.render() # Display pop up window
        action = 0 if np.matmul(weights, observation) < 0 else 1 # gets the action index; action space size is 2
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    episodes_array = []
    for round in range(1000):
        best_weights, best_score = np.random.uniform(-1, 1, size=4), 0
        for episode_num in range(1,10001):
            weight = np.random.uniform(-1, 1, size=4)
            score = run_episode(env=env, weights=weight)
            best_weights = weight if score > best_score else best_weights
            if score >= 200:
                break
        episodes_array.append(episode_num)
    fig = plt.figure()
    plt.hist(np.array(episodes_array), bins=len(set(episodes_array)))
    # plt.xticks(range(1, np.max(episodes_array))[::15])
    plt.title('Histogram of Num of Episodes to Get 200 Pts.')
    # plt.show()
    fig.savefig('Num_episodes.png')
    plt.close()
    env.close()
    print(f'Average number of episodes is: {np.average(episodes_array):.2f}')
