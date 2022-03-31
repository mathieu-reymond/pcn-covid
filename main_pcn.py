import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import wandb


class MultiDiscreteAction(gym.ActionWrapper):
    def __init__(self, env, action_map):
        super(MultiDiscreteAction, self).__init__(env)
        self.action_map = action_map
        self.action_space = gym.spaces.MultiDiscrete([len(am) for am in action_map])

    def action(self, action):
        return np.array([self.action_map[i][action[i]] for i in range(len(self.action_map))])


class ScaleRewardEnv(gym.RewardWrapper):
    def __init__(self, env, min_=0., scale=1.):
        gym.RewardWrapper.__init__(self, env)
        self.min = min_
        self.scale = scale

    def reward(self, reward):
        return (reward - self.min)/self.scale


class TodayWrapper(gym.Wrapper):
    def __init__(self, env):
        super(TodayWrapper, self).__init__(env)

    def reset(self):
        s = super(TodayWrapper, self).reset()
        ss, se, sa = s
        return (ss[-1].T, se[-1], sa)
    # step function of covid env returns simulation results of every day of timestep
    # only keep current day
    # also discard first reward
    def step(self, action):
        s, r, d, i = super(TodayWrapper, self).step(action)
        ss, se, sa = s
        # sum all the social burden objectives together:
        p_tot = r[2:].sum()
        r = np.concatenate((r, p_tot[None]))
        return (ss[-1].T, se[-1], sa), r, d, i


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


ss_emb = {
    'conv1d': nn.Sequential(
            nn.Conv1d(10, 20, kernel_size=3, stride=2, groups=5),
            nn.ReLU(),
            nn.Conv1d(20, 20, kernel_size=2, stride=1, groups=10),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(100, 64),
            nn.Sigmoid()
        ),
    'small': nn.Sequential(
            nn.Flatten(),
            nn.Linear(130, 64),
            nn.Sigmoid()
        ),
    'big': nn.Sequential(
            nn.Flatten(),
            nn.Linear(130, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Sigmoid()
        ),
}


se_emb = {
    'small': nn.Sequential(
            nn.Linear(1, 64),
            nn.Sigmoid()
        ),
    'big': nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Sigmoid()
        )
}


sa_emb = {
    'small': nn.Sequential(
            nn.Linear(3, 64),
            nn.Sigmoid()
        ),
    'big': nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Sigmoid()
        )
}

class CovidModel(nn.Module):

    def __init__(self,
                 nA,
                 scaling_factor,
                 objectives,
                 ss_emb,
                 se_emb,
                 sa_emb):
        super(CovidModel, self).__init__()

        self.scaling_factor = scaling_factor[:,objectives + (len(scaling_factor)-1,)]
        self.objectives = objectives
        self.ss_emb = ss_emb
        self.se_emb = se_emb
        self.sa_emb = sa_emb
        self.s_emb = nn.Sequential(
            nn.Linear(64, 64),
            nn.Sigmoid()
        )
        self.c_emb = nn.Sequential(nn.Linear(self.scaling_factor.shape[-1], 64),
                                   nn.Sigmoid())
        self.fc = nn.Sequential(nn.Linear(64, 64),
                                nn.ReLU(),
                                nn.Linear(64, nA))

    def forward(self, state, desired_return, desired_horizon):
        # filter desired_return to only keep used objectives
        desired_return = desired_return[:,self.objectives]
        c = torch.cat((desired_return, desired_horizon), dim=-1)
        # commands are scaled by a fixed factor
        c = c*self.scaling_factor
        ss, se, sa = state
        s = self.ss_emb(ss.float())*self.se_emb(se.float())*self.sa_emb(sa.float())
        s = self.s_emb(s)
        c = self.c_emb(c)
        # element-wise multiplication of state-embedding and command
        log_prob = self.fc(s*c)
        return log_prob


class DiscreteHead(nn.Module):
    def __init__(self, base):
        super(DiscreteHead, self).__init__()
        self.base = base
    def forward(self, state, desired_return, desired_horizon):
        x = self.base(state, desired_return, desired_horizon)
        x = F.log_softmax(x, 1)
        return x


class MultiDiscreteHead(nn.Module):
    def __init__(self, base):
        super(MultiDiscreteHead, self).__init__()
        self.base = base
    def forward(self, state, desired_return, desired_horizon):
        x = self.base(state, desired_return, desired_horizon)
        b, o = x.shape
        # hardcoded
        x = x.reshape(b, 3, 3)
        x = F.log_softmax(x, 2)
        return x


class ContinuousHead(nn.Module):
    def __init__(self, base):
        super(ContinuousHead, self).__init__()
        self.base = base
    def forward(self, state, desired_return, desired_horizon):
        x = self.base(state, desired_return, desired_horizon)
        x = torch.sigmoid(x)
        # bound in [0, 1]
        # x = (x+1)/2
        return x


def multidiscrete_env(env):
    # the discrete actions:
    a = [[0.0, 0.3, 0.6],[0, 0.5, 1],[0.3, 0.6, 0.9]]
    env = MultiDiscreteAction(env, a)
    return env


if __name__ == '__main__':
    import torch
    import argparse
    from pcn import train
    from datetime import datetime
    import uuid
    import os
    import gym_covid

    parser = argparse.ArgumentParser(description='PCN')
    parser.add_argument('--objectives', default=[1, 5], type=int, nargs='+', help='index for ari, arh, pw, ps, pl, ptot')
    parser.add_argument('--env', default='ode', type=str, help='ode or binomial')
    parser.add_argument('--action', default='discrete', type=str, help='discrete, multidiscrete or continuous')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--steps', default=3e5, type=float, help='total timesteps')
    parser.add_argument('--batch', default=256, type=int, help='batch size')
    parser.add_argument('--model-updates', default=50, type=int,
        help='number of times the model is updated at every training iteration')
    parser.add_argument('--top-episodes', default=200, type=int,
        help='top-n episodes used to compute target-return and horizon. \
              Initially fill ER with n random episodes')
    parser.add_argument('--n-episodes', default=10, type=int,
        help='number of episodes to run between each training iteration')
    parser.add_argument('--er-size', default=400, type=int,
        help='max size (in episodes) of the ER buffer')
    parser.add_argument('--threshold', default=0.02, type=float, help='crowding distance threshold before penalty')
    parser.add_argument('--noise', default=0.0, type=float, help='noise applied on target-return on batch-update')
    parser.add_argument('--model', default='conv1dsmall', type=str, help='conv1d(big|small), dense(big|small)')
    args = parser.parse_args()
    print(args)

    device = 'cpu'

    env_type = 'ODE' if args.env == 'ode' else 'Binomial'
    n_evaluations = 1 if env_type == 'ODE' else 10
    scale = np.array([800000, 10000, 50., 20, 50, 100])
    ref_point = np.array([-15000000, -200000, -1000.0, -1000.0, -1000.0, -1000.0])/scale
    scaling_factor = torch.tensor([[1, 1, 1, 1, 1, 1, 0.1]]).to(device)
    max_return = np.array([0, 0, 0, 0, 0, 0])/scale
    # max_return = np.array([0, -8000, 0, 0, 0, 0])/scale
    # keep only a selection of objectives

    if args.action == 'discrete':
        env = gym.make(f'BECovidWithLockdown{env_type}Discrete-v0')
        nA = env.action_space.n
    else:
        env = gym.make(f'BECovidWithLockdown{env_type}Continuous-v0')
        if args.action == 'multidiscrete':
            env = multidiscrete_env(env)
            nA = env.action_space.nvec.sum()
        # continuous
        else:
            nA = np.prod(env.action_space.shape)
    env = TodayWrapper(env)
    env = ScaleRewardEnv(env, scale=scale)

    env.nA = nA
    
    if args.model == 'conv1dbig':
        ss, se, sa = ss_emb['conv1d'], se_emb['big'], sa_emb['big']
    elif args.model == 'conv1dsmall':
        ss, se, sa = ss_emb['conv1d'], se_emb['small'], sa_emb['small']
    elif args.model == 'densebig':
        ss, se, sa = ss_emb['big'], se_emb['big'], sa_emb['big']
    elif args.model == 'densesmall':
        ss, se, sa = ss_emb['small'], se_emb['small'], sa_emb['small']
    else:
        raise ValueError(f'unknown model type: {args.model}')
    model = CovidModel(nA, scaling_factor, tuple(args.objectives), ss, se, sa).to(device)

    if args.action == 'discrete':
        model = DiscreteHead(model)
    elif args.action == 'multidiscrete':
        model = MultiDiscreteHead(model)
    elif args.action == 'continuous':
        model = ContinuousHead(model)
    # if args.model is not None:
    #     model = torch.load(args.model, map_location=device).to(device)
    #     model.scaling_factor = model.scaling_factor.to(device)

    logdir = f'{os.getenv("VSC_SCRATCH", "/tmp")}/pcn/commit_4bb5737a95d81615ce00b8a3b06f5d3197007627/'
    logdir += '/'.join([f'{k}_{v}' for k, v in vars(args).items()]) + '/'
    logdir += datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str(uuid.uuid4())[:4] + '/'

    wandb.init(project='pcn-covid', entity='mreymond', config={k: v for k, v in vars(args).items()})

    train(env,
        model,
        learning_rate=args.lr,
        batch_size=args.batch,
        total_steps=args.steps,
        n_model_updates=args.model_updates,
        n_er_episodes=args.top_episodes,
        n_step_episodes=args.n_episodes,
        max_size=args.er_size,
        max_return=max_return,
        threshold=args.threshold,
        ref_point=ref_point,
        noise=args.noise,
        n_evaluations=n_evaluations,
        objectives=tuple(args.objectives),
        logdir=logdir)
