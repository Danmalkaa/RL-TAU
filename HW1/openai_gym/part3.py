import gym
import numpy as np


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
    weight = np.random.uniform(-1, 1, size=4)
    score = run_episode(env=env, weights=weight)
    env.close()
    print(f'Total Reward is: {score}')
