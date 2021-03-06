"""
    This file is copied/apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""
import sys
import pickle
import numpy as np
from collections import namedtuple
from itertools import count
import random
import gym.spaces
import os
import time
import string
import random
from scipy import stats


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from utils.replay_buffer import ReplayBuffer, sample_n_unique
from utils.gym import get_wrapper_by_name

STATS_SAVE_PATH = "/kaggle/working/RL_EXP/Project"
FILE_NAME ="stats_"+time.strftime("%Y%m%d-%H%M%S")+"_"+(''.join(random.choice(string.ascii_letters) for _ in range(5)))
FULL_PATH  = STATS_SAVE_PATH+FILE_NAME+'.pkl'
print("Saving file output to "+STATS_SAVE_PATH)
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

"""
    OptimizerSpec containing following attributes
        constructor: The optimizer constructor ex: RMSprop
        kwargs: {Dict} arguments for constructing optimizer
"""
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

Statistic = {
    "mean_episode_rewards": [],
    "best_mean_episode_rewards": []
}

def dqn_learing(
    env,
    q_func,
    optimizer_spec,
    exploration,
    stopping_criterion=None,
    replay_buffer_size=1000000,
    batch_size=32,
    gamma=0.99,
    learning_starts=50000,
    learning_freq=4,
    frame_history_len=4,
    target_update_freq=10000,
    dynamic_exp_model=None
    ):

    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            input_channel: int
                number of channel of input.
            num_actions: int
                number of actions
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    exploration: Schedule (defined in utils.schedule)
        schedule for probability of chosing random action.
    stopping_criterion: (env) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    ###############
    # BUILD MODEL #
    ###############

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_arg = env.observation_space.shape[0]
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_arg = frame_history_len * img_c
    num_actions = env.action_space.n

    # Construct an epilson greedy policy with given exploration schedule
    def select_epilson_greedy_action(model, obs, t):
        sample = random.random()
        eps_threshold = exploration.value(t)
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) / 255.0
            # with torch.no_grad() variable is only used in inference mode, i.e. don???t save the history
            with torch.no_grad():
                return model(Variable(obs, volatile=True)).data.max(1)[1].cpu()
        else:
            return torch.IntTensor([[random.randrange(num_actions)]])

    def select_guided_explore_action(model, obs, t, explore_kwargs):
        sample = random.random()
        eps_threshold = exploration.value(t)
        eps_threshold = max(0.05, eps_threshold)
        if sample > eps_threshold:
            obs = torch.from_numpy(obs).type(dtype).unsqueeze(0) / 255.0
            # with torch.no_grad() variable is only used in inference mode, i.e. don???t save the history
            with torch.no_grad():
                return model(Variable(obs, volatile=True)).data.max(1)[1].cpu()
        else:
            with torch.no_grad():
                return explore(**explore_kwargs)

    def get_probability(state, samples):
        design = samples
        cov = design.cov()
        mean = design.mean(dim=1)
        try:
            p = stats.multivariate_normal.pdf(state.detach().numpy(), mean.detach().numpy(), cov.detach().numpy())
        except:
            p = 1.0
        return p

    def explore(d_model, state, replay_memory, num_actions):
        d_model.eval()
        N = replay_buffer_size
        num_samples = 50
        samples = []
        # idxes = sample_n_unique(lambda: range(N-num_samples,N), num_samples)
        _, act_batch, _, _, _ = replay_memory._encode_sample(list(range(N-num_samples-1,N-1)))
        act_batch_t = torch.from_numpy(act_batch).float().unsqueeze(1).to(device)
        for i in range(N-num_samples-2,N-2):
            outputs = d_model(torch.from_numpy(replay_memory._encode_observation((i - 1) % replay_memory.size)).unsqueeze(0).float().to(device) / 255.0)
            samples.append(outputs)
        samples = torch.stack(samples).squeeze()
        samples = torch.concat((samples,act_batch_t),dim=1)

        least_p = np.inf
        best_a = -1
        next_state = d_model(torch.from_numpy(state).unsqueeze(0).float().to(device))
        for action in range(num_actions):
            # next_state = d_model(np.append(state, [[[action]]], axis=1))
            tmp_next_state = torch.concat((next_state,torch.Tensor([float(action)]).unsqueeze(0).to(device)),dim=1)
            p = get_probability(tmp_next_state, samples)
            if p < least_p:
                best_a = action
                least_p = p
        d_model.train()
        return best_a


    def fit_dynamics_model(model,optim,loss,samples):
        obs_batch_t, act_batch_t, _, next_obs_batch_t, _ = samples
        batched_inputs = obs_batch_t
        batched_output = model(batched_inputs)
        batched_output = torch.concat((batched_output,act_batch_t),dim=1)
        with torch.no_grad():
            batched_targets = model(next_obs_batch_t)
            batched_targets = torch.concat((batched_targets,act_batch_t),dim=1)
        batch_loss = loss(batched_targets,batched_output)
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

    # Initialize target q function and q function, i.e. build the model.
    ######
    if torch.cuda.is_available():
      print("running on CUDA")
      device = torch.device("cuda")
      Q = q_func(input_arg,num_actions).to(device)
      target_Q = q_func(input_arg,num_actions).to(device)
      if dynamic_exp_model is not None:
          D_exp_model = dynamic_exp_model(input_arg,num_actions).to(device)
    else:
      print("running on CPU")
      device = torch.device("cpu")
      Q = q_func(input_arg,num_actions)
      target_Q = q_func(input_arg,num_actions)
      if dynamic_exp_model is not None:
          D_exp_model = dynamic_exp_model(input_arg,num_actions)
    if dynamic_exp_model is not None:
        D_criterion = nn.MSELoss(reduction='sum')
        D_opt_spec = OptimizerSpec(
            constructor=torch.optim.Adam,
            kwargs=dict(lr=0.1))
        D_opt = D_opt_spec.constructor(D_exp_model.parameters(), **D_opt_spec.kwargs)
    ######


    # Construct Q network optimizer function
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # Construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 10000
    learn_start = False

    for t in count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env):
            break

        ### 2. Step the env and store the transition
        # At this point, "last_obs" contains the latest observation that was
        # recorded from the simulator. Here, your code needs to store this
        # observation and its outcome (reward, next observation, etc.) into
        # the replay buffer while stepping the simulator forward one step.
        # At the end of this block of code, the simulator should have been
        # advanced one step, and the replay buffer should contain one more
        # transition.
        # Specifically, last_obs must point to the new latest observation.
        # Useful functions you'll need to call:
        # obs, reward, done, info = env.step(action)
        # this steps the environment forward one step
        # obs = env.reset()
        # this resets the environment if you reached an episode boundary.
        # Don't forget to call env.reset() to get a new observation if done
        # is true!!
        # Note that you cannot use "last_obs" directly as input
        # into your network, since it needs to be processed to include context
        # from previous frames. You should check out the replay buffer
        # implementation in dqn_utils.py to see what functionality the replay
        # buffer exposes. The replay buffer has a function called
        # encode_recent_observation that will take the latest observation
        # that you pushed into the buffer and compute the corresponding
        # input that should be given to a Q network by appending some
        # previous frames.
        # Don't forget to include epsilon greedy exploration!
        # And remember that the first time you enter this loop, the model
        # may not yet have been initialized (but of course, the first step
        # might as well be random, since you haven't trained your net...)
        #####

        last_obs_index= replay_buffer.store_frame(last_obs)
        encoded_recent_obs = replay_buffer.encode_recent_observation() /255.0
        if dynamic_exp_model is not None:
            explore_kwargs_dict = dict(d_model=D_exp_model, state=encoded_recent_obs, replay_memory=replay_buffer, num_actions=num_actions)
            chosen_action = select_guided_explore_action(Q,encoded_recent_obs, t, explore_kwargs_dict) if learn_start else select_epilson_greedy_action(Q,encoded_recent_obs, 0)
        else:
            chosen_action = select_epilson_greedy_action(Q, encoded_recent_obs, t)
        obs, reward, is_done, details = env.step(chosen_action)
        replay_buffer.store_effect(last_obs_index, chosen_action, reward, is_done)

        if is_done:
            obs = env.reset()
        last_obs = obs

        #####

        # at this point, the environment should have been advanced one step (and
        # reset if done was true), and last_obs should point to the new latest
        # observation

        ### 3. Perform experience replay and train the network.
        # Note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):
            # Here, you should perform training. Training consists of four steps:
            # 3.a: use the replay buffer to sample a batch of transitions (see the
            # replay buffer code for function definition, each batch that you sample
            # should consist of current observations, current actions, rewards,
            # next observations, and done indicator).
            # Note: Move the variables to the GPU if avialable
            # 3.b: fill in your own code to compute the Bellman error. This requires
            # evaluating the current and next Q-values and constructing the corresponding error.
            # Note: don't forget to clip the error between [-1,1], multiply is by -1 (since pytorch minimizes) and
            #       maskout post terminal status Q-values (see ReplayBuffer code).
            # 3.c: train the model. To do this, use the bellman error you calculated perviously.
            # Pytorch will differentiate this error for you, to backward the error use the following API:
            #       current.backward(d_error.data.unsqueeze(1))
            # Where "current" is the variable holding current Q Values and d_error is the clipped bellman error.
            # Your code should produce one scalar-valued tensor.
            # Note: don't forget to call optimizer.zero_grad() before the backward call and
            #       optimizer.step() after the backward call.
            # 3.d: periodically update the target network by loading the current Q network weights into the
            #      target_Q network. see state_dict() and load_state_dict() methods.
            #      you should update every target_update_freq steps, and you may find the
            #      variable num_param_updates useful for this (it was initialized to 0)
            #####

            learn_start = True
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # get samples batch
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)


            # calc Q values
            obs_batch_t = torch.from_numpy(obs_batch).float().to(device) / 255
            next_obs_batch_t = torch.from_numpy(next_obs_batch).float().to(device) /255
            rew_batch_t = torch.from_numpy(rew_batch).float().to(device)
            act_batch_t = torch.from_numpy(act_batch).long().unsqueeze(1).to(device)
            not_done_mask = Variable(torch.from_numpy(1 - done_mask)).type(dtype)

            # Train Dynamic Network
            if num_param_updates % 100 == 0 and dynamic_exp_model is not None:
                samples = (obs_batch_t, act_batch_t, rew_batch_t, next_obs_batch_t, done_mask)
                fit_dynamics_model(D_exp_model,D_opt,D_criterion,samples)

            Q_vals = Q(obs_batch_t).gather(1, index=act_batch_t).to(device).squeeze()

            # calc next Q values
            Q_next = torch.zeros(batch_size, device=device)
            Q_next = target_Q(next_obs_batch_t).detach().max(1)[0]
            next_Q_values = not_done_mask * Q_next

            # calculate loss
            expected_vals = torch.zeros(batch_size, device=device)
            expected_vals = rew_batch_t + (gamma * next_Q_values)
            # loss_func = nn.SmoothL1Loss(reduction='none')
            # error = loss_func(expected_vals, Q_vals).clamp(-1, 1) * -1
            error = expected_vals-Q_vals
            clip = error.clamp(-1, 1) * -1.0


            # update model
            optimizer.zero_grad()
            Q_vals.backward(clip.data) ## clip.data.unsqueeze(1)
            optimizer.step()

            # update target model
            num_param_updates += 1
            if num_param_updates % target_update_freq == 0:
                target_Q.load_state_dict(Q.state_dict())

            #####
        ### 4. Log progress and keep track of statistics
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

        Statistic["mean_episode_rewards"].append(mean_episode_reward)
        Statistic["best_mean_episode_rewards"].append(best_mean_episode_reward)

        if t % LOG_EVERY_N_STEPS == 0 and t > learning_starts:
            print("Timestep %d" % (t,))
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            sys.stdout.flush()

            # Dump statistics to pickle
            with open(FULL_PATH, 'wb') as f:
                pickle.dump(Statistic, f)
                print("Saved to %s" % FILE_NAME)
