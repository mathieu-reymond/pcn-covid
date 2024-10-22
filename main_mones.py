import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import copy


ss_emb = {
    'conv1d': nn.Sequential(
            nn.Conv1d(10, 20, kernel_size=3, stride=2, groups=5),
            nn.ReLU(),
            nn.Conv1d(20, 20, kernel_size=2, stride=1, groups=10),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(100, 64),
            nn.SiLU()
        ),
    'small': nn.Sequential(
            nn.Flatten(),
            nn.Linear(130, 64),
            nn.Sigmoid()
        ),
    'big': nn.Sequential(
            nn.Flatten(),
            nn.Linear(130, 64),
            nn.SiLU(),
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
            nn.SiLU(),
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
            nn.SiLU(),
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
                 sa_emb,
                 with_budget=False):
        super(CovidModel, self).__init__()

        self.scaling_factor = scaling_factor[:,objectives + (len(scaling_factor)-1,)]
        self.objectives = objectives
        self.ss_emb = ss_emb
        self.se_emb = se_emb
        self.sa_emb = sa_emb
        if with_budget:
            # sa_emb has 3 inputs (1 per action), same as budget (1 budget per action)
            self.sb_emb = copy.deepcopy(sa_emb)
        else:
            # otherwise, we include the number of steps left (single int),
            # se_emb takes a single input as well
            self.sb_emb = copy.deepcopy(se_emb)
        self.s_emb = nn.Sequential(
            nn.Linear(64, 64),
            nn.SiLU()
        )
        self.c_emb = nn.Sequential(nn.Linear(self.scaling_factor.shape[-1], 64),
                                   nn.SiLU())
        self.fc = nn.Sequential(nn.Linear(64, 64),
                                nn.SiLU(),
                                nn.Linear(64, nA))

    def forward(self, state):
        # if self.sb_emb is not None:
        sb, ss, se, sa = state
        s = self.ss_emb(ss.float())*self.se_emb(se.float())*self.sa_emb(sa.float())*self.sb_emb(sb.float())
        # sc = torch.cat((s, c), 1)
        log_prob = self.fc(s)
        return log_prob


class DiscreteHead(nn.Module):
    def __init__(self, base):
        super(DiscreteHead, self).__init__()
        self.base = base
    def forward(self, state):
        x = self.base(state)
        x = F.log_softmax(x, 1)
        return x


class MultiDiscreteHead(nn.Module):
    def __init__(self, base):
        super(MultiDiscreteHead, self).__init__()
        self.base = base
    def forward(self, state):
        x = self.base(state)
        b, o = x.shape
        # hardcoded
        x = x.reshape(b, 3, 3)
        x = F.log_softmax(x, 2)
        return x


class ContinuousHead(nn.Module):
    def __init__(self, base):
        super(ContinuousHead, self).__init__()
        self.base = base
    def forward(self, state):
        x = self.base(state)
        x = torch.sigmoid(x)
        # bound in [0, 1]
        # x = (x+1)/2
        return x


if __name__ == '__main__':
    import torch
    from gym.wrappers import TimeLimit
    import argparse
    from mones.mones import MONES
    from datetime import datetime
    import uuid
    import os
    import gym_covid
    from main_pcn import multidiscrete_env, TodayWrapper, ScaleRewardEnv

    parser = argparse.ArgumentParser(description='MONES')
    parser.add_argument('--objectives', default=[1, 5], type=int, nargs='+', help='index for ari, arh, pw, ps, pl, ptot')
    parser.add_argument('--env', default='ode', type=str, help='ode or binomial')
    parser.add_argument('--action', default='discrete', type=str, help='discrete, multidiscrete or continuous')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--model', default='conv1dsmall', type=str, help='conv1d(big|small), dense(big|small)')
    parser.add_argument('--budget', default=None, type=int, help='number of times each action is allowed to change')
    parser.add_argument('--population', default=200, type=int, help='pop size')
    parser.add_argument('--indicator', default='hypervolume', type=str)
    parser.add_argument('--procs', default=1, type=int, help='parallel runs')
    args = parser.parse_args()

    device = 'cpu'

    env_type = 'ODE' if args.env == 'ode' else 'Binomial'
    n_evaluations = 1 if env_type == 'ODE' else 10
    budget = f'Budget{args.budget}' if args.budget is not None else ''
    scale = np.array([800000, 11000, 50., 20, 50, 120])
    ref_point = np.array([-15000000, -200000, -1000.0, -1000.0, -1000.0, -1000.0])/scale
    scaling_factor = torch.tensor([[1, 1, 1, 1, 1, 1, 0.1]]).to(device)
    max_return = np.array([0, 0, 0, 0, 0, 0])/scale
    n_runs = 1
    train_iterations = 1000
    # max_return = np.array([0, -8000, 0, 0, 0, 0])/scale
    # keep only a selection of objectives

    def make_env():
        if args.action == 'discrete':
            env = gym.make(f'BECovidWithLockdown{env_type}Discrete-v0')
            nA = env.action_space.n
        else:
            env = gym.make(f'BECovidWithLockdown{env_type}{budget}Continuous-v0')
            if args.action == 'multidiscrete':
                env = multidiscrete_env(env)
                nA = env.action_space.nvec.sum()
            # continuous
            else:
                nA = np.prod(env.action_space.shape)
        env = TodayWrapper(env)
        env = ScaleRewardEnv(env, scale=scale)

        env.nA = nA
        return env
    
    nA = make_env().nA  
    
    if args.model == 'conv1dbig':
        ss, se, sa = ss_emb['conv1d'], se_emb['big'], sa_emb['big']
    elif args.model == 'conv1dsmall':
        ss, se, sa = ss_emb['conv1d'], se_emb['small'], sa_emb['small']
    elif args.model.startswith('densebig'):
        ss, se, sa = ss_emb['big'], se_emb['big'], sa_emb['big']
    elif args.model == 'densesmall':
        ss, se, sa = ss_emb['small'], se_emb['small'], sa_emb['small']
    else:
        raise ValueError(f'unknown model type: {args.model}')
    with_budget = args.budget is not None
    model = CovidModel(nA, scaling_factor, tuple(args.objectives), ss, se, sa, with_budget=with_budget).to(device)

    if args.action == 'discrete':
        model = DiscreteHead(model)
    elif args.action == 'multidiscrete':
        model = MultiDiscreteHead(model)
    elif args.action == 'continuous':
        model = ContinuousHead(model)

    # if args.model is not None:
    #     model = torch.load(args.model).to(device)

    logdir = f'{os.getenv("LOGDIR","/tmp")}/pcn/mones/{args.env}/{args.indicator}/lr_{args.lr}/population_{args.population}/runs_{n_runs}/train_iterations_{train_iterations}/'
    logdir += datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str(uuid.uuid4())[:4] + '/'

    agent = MONES(
        make_env,
        model,
        n_population=args.population,
        n_runs=n_runs,
        ref_point=ref_point[args.objectives],
        lr=args.lr,
        indicator=args.indicator,
        logdir=logdir,
        objectives=args.objectives,
        n_processes=args.procs,
    )
    agent.train(train_iterations)
