import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class ScaleRewardEnv(gym.RewardWrapper):
    def __init__(self, env, min_=0., scale=1.):
        gym.RewardWrapper.__init__(self, env)
        self.min = min_
        self.scale = scale

    def reward(self, reward):
        return (reward - self.min)/self.scale


class TodayWrapper(gym.Wrapper):
    def reset(self):
        s = super(TodayWrapper, self).reset()
        return s[-1].T
    # step function of covid env returns simulation results of every day of timestep
    # only keep current day
    # also discard first reward
    def step(self, action):
        s, r, d, i = super(TodayWrapper, self).step(action)
        return s[-1].T, r[1:], d, i


class HistoryEnv(gym.Wrapper):
    def __init__(self, env, size=4):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.size = size
        # will be set in _convert
        self._state = None

        # history stacks observations on dim 0
        low = np.repeat(self.observation_space.low, self.size, axis=0)
        high = np.repeat(self.observation_space.high, self.size, axis=0)
        self.observation_space = gym.spaces.Box(low, high)

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        state = self.env.reset(**kwargs)
        # add history dimension
        s = np.expand_dims(state, 0)
        # fill history with current state
        self._state = np.repeat(s, self.size, axis=0)
        return np.concatenate(self._state, axis=0)

    def step(self, ac):
        state, r, d, i = self.env.step(ac)
        # shift history
        self._state = np.roll(self._state, -1, axis=0)
        # add state to history
        self._state[-1] = state
        return np.concatenate(self._state, axis=0), r, d, i


class CovidModel(nn.Module):

    def __init__(self, nA, scaling_factor, n_hidden=64):
        super(CovidModel, self).__init__()

        self.scaling_factor = scaling_factor
        self.s_emb = nn.Sequential(
            nn.Conv1d(10, 20, kernel_size=3, stride=2, groups=5),
            nn.ReLU(),
            nn.Conv1d(20, 20, kernel_size=2, stride=1, groups=10),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(100, 64),
            nn.Sigmoid()
        )
        self.c_emb = nn.Sequential(nn.Linear(3, 64),
                                   nn.Sigmoid())
        self.fc = nn.Sequential(nn.Linear(64, 64),
                                nn.ReLU(),
                                nn.Linear(64, nA),
                                nn.LogSoftmax(1))

    def forward(self, state, desired_return, desired_horizon):
        c = torch.cat((desired_return, desired_horizon), dim=-1)
        # commands are scaled by a fixed factor
        c = c*self.scaling_factor
        s = self.s_emb(state.float())
        c = self.c_emb(c)
        # element-wise multiplication of state-embedding and command
        log_prob = self.fc(s*c)
        return log_prob


if __name__ == '__main__':
    import torch
    import argparse
    from pcn import train
    from datetime import datetime
    import uuid
    import os
    import gym_covid

    parser = argparse.ArgumentParser(description='PCN')
    parser.add_argument('--env', default='covid', type=str, help='covid')
    parser.add_argument('--model', default=None, type=str, help='load model')
    args = parser.parse_args()

    device = 'cpu'

    if args.env == 'covid':
        scale = np.array([10000, 100.])
        env = gym.make('BECovidWithLockdownODEDiscrete-v0')
        env = TodayWrapper(env)
        env = ScaleRewardEnv(env, scale=scale)
        nA = env.action_space.n
        ref_point = np.array([-50000, -2000.0])/scale
        scaling_factor = torch.tensor([[1, 1, 0.1]]).to(device)
        max_return = np.array([-8000, 0])/scale

        model = CovidModel(nA, scaling_factor).to(device)
        lr, total_steps, batch_size, n_model_updates, n_er_episodes, max_size = 1e-3, 2e6, 256, 50, 50, 200

    env.nA = nA

    if args.model is not None:
        model = torch.load(args.model, map_location=device).to(device)
        model.scaling_factor = model.scaling_factor.to(device)

    logdir = f'{os.getenv("LOGDIR", "/tmp")}/pcn/pcn/{args.env}/lr_{lr}/totalsteps_{total_steps}/batch_size_{batch_size}/n_model_updates_{n_model_updates}/n_er_episodes_{n_er_episodes}/max_size_{max_size}/'
    logdir += datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str(uuid.uuid4())[:4] + '/'

    train(env,
        model,
        learning_rate=lr,
        batch_size=batch_size,
        total_steps=total_steps,
        n_model_updates=n_model_updates,
        n_er_episodes=n_er_episodes,
        max_size=max_size,
        max_return=max_return,
        ref_point=ref_point,
        logdir=logdir)