from main_pcn import CovidModel, CovidModel2, MultiDiscreteHead, DiscreteHead, ContinuousHead, ScaleRewardEnv, TodayWrapper, multidiscrete_env
from pcn import non_dominated, Transition, choose_action
import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime


device = 'cpu'


def run_episode(env, model, desired_return, desired_horizon, max_return):
    transitions = []
    obs = env.reset()
    done = False
    while not done:
        action = choose_action(model, obs, desired_return, desired_horizon, eval=True)
        n_obs, reward, done, _ = env.step(action)

        transitions.append(Transition(
            observation=obs[0],
            action=env.action(action),
            reward=np.float32(reward).copy(),
            next_observation=n_obs,
            terminal=done
        ))

        obs = n_obs
        # clip desired return, to return-upper-bound, 
        # to avoid negative returns giving impossible desired returns
        desired_return = np.clip(desired_return-reward, None, max_return, dtype=np.float32)
        # clip desired horizon to avoid negative horizons
        desired_horizon = np.float32(max(desired_horizon-1, 1.))
    return transitions


def plot_episode(transitions, alpha=1.):
    states = np.array([t.observation for t in transitions])
    i_hosp_new = states[...,-3].sum(axis=-1)
    i_icu_new = states[...,-2].sum(axis=-1)
    d_new = states[...,-1].sum(axis=-1)
    actions = np.array([t.action for t in transitions])

    # steps in dates
    start = datetime.date(2020, 5, 3)
    week = datetime.timedelta(days=7)
    dates = [start+week*i for i in range(0, 18, 2)]
    
    axs = plt.gcf().axes
    # hospitalizations
    ax = axs[0]
    ax.plot(i_hosp_new, alpha=alpha, label='hosp', color='blue')
    ax.plot(i_icu_new,  alpha=alpha, label='icu', color='green')
    ax.plot(i_hosp_new+i_icu_new, label='hosp+icu',  alpha=alpha, color='orange')
    ax.set_xticks(ticks=np.arange(0, 18, 2), labels=[str(d.day)+'/'+str(d.month) for d in dates])

    # deaths
    ax = axs[1]
    ax.plot(d_new, alpha=alpha, label='deaths', color='red')

    # actions
    ax = axs[2]
    ax.set_ylim([0, 1])
    ax.plot(actions[:,0], alpha=alpha, label='p_w', color='blue')
    ax.plot(actions[:,1], alpha=alpha, label='p_s', color='orange')
    ax.plot(actions[:,2], alpha=alpha, label='p_l', color='green')

    axs[0].set_xlabel('days')
    axs[0].set_ylabel('hospitalizations')
    axs[1].set_ylabel('deaths')
    axs[2].set_ylabel('actions')
    # for ax in axs:
    #     ax.legend()


def eval_pcn(env, model, desired_return, desired_horizon, max_return, gamma=1., n=1):
    plt.subplots(3, 1, sharex=True)
    alpha = 1 if n == 1 else 0.2
    returns = np.empty((n, desired_return.shape[-1]))
    for n_i in range(n):
        transitions = run_episode(env, model, desired_return, desired_horizon, max_return)
        # compute return
        for i in reversed(range(len(transitions)-1)):
            transitions[i].reward += gamma * transitions[i+1].reward
        
        returns[n_i] = transitions[i].reward.flatten()
        print(f'ran model with desired-return: {desired_return.flatten()}, got {transitions[i].reward.flatten()}')
        print('action sequence: ')
        for t in transitions:
            print(f'- {t.action}')
        plot_episode(transitions, alpha)
    print(f'ran model with desired-return: {desired_return.flatten()}, got average return {returns.mean(0).flatten()}')


if __name__ == '__main__':
    import argparse
    import uuid
    import os
    import gym_covid
    import gym
    import pathlib
    import h5py

    parser = argparse.ArgumentParser(description='PCN')
    parser.add_argument('env', type=str, help='ode or binomial')
    parser.add_argument('model', type=str, help='load model')
    parser.add_argument('--n', type=int, default=1, help='evaluation runs')
    parser.add_argument('--interactive', action='store_true', help='interactive policy selection')
    parser.set_defaults(interactive=False)
    args = parser.parse_args()
    model_dir = pathlib.Path(args.model)

    log = model_dir / 'log.h5'
    log = h5py.File(log)
    checkpoints = [str(p) for p in model_dir.glob('model_*.pt')]
    checkpoints = sorted(checkpoints)
    model = torch.load(checkpoints[-1])

    with log:
        pareto_front = log['train/leaves/ndarray'][-1]
        pareto_front = non_dominated(pareto_front)
        pf = np.argsort(pareto_front, axis=0)
        pareto_front = pareto_front[pf[:,0]]
        
    env_type = 'ODE' if args.env == 'ode' else 'Binomial'
    scale = np.array([10000, 100.])
    # hacky, d for discrete, m for multidiscrete, c for continuous
    action = str(model_dir)[str(model_dir).find('action')+7]
    if action == 'd':
        env = gym.make(f'BECovidWithLockdown{env_type}Discrete-v0')
        nA = env.action_space.n
    else:
        env = gym.make(f'BECovidWithLockdown{env_type}Continuous-v0')
        if action == 'm':
            env = multidiscrete_env(env)
            nA = env.action_space.nvec.sum()
        else:
            nA = np.prod(env.action_space.shape)
            env.action = lambda x: x
    env = TodayWrapper(env)
    env = ScaleRewardEnv(env, scale=scale)
    ref_point = np.array([-50000, -2000.0])/scale
    scaling_factor = torch.tensor([[1, 1, 0.1]])
    max_return = np.array([-8000, 0])/scale
    print(env)

    inp = -1
    if not args.interactive:
        (model_dir / 'policies-executions').mkdir(exist_ok=True)
        print('='*38)
        print('not interactive, this may take a while')
        print('='*38)
    while True:
        if args.interactive:
            print('solutions: ')
            for i, p in enumerate(pareto_front):
                print(f'{i} : {p}')
            inp = input('-> ')
            inp = int(inp)
        else:
            inp = inp+1
            if inp >= len(pareto_front):
                break
        desired_return = pareto_front[inp]
        desired_horizon = 17

        eval_pcn(env, model, desired_return, desired_horizon, max_return, n=args.n)
        if args.interactive:
            plt.show()
        else:
            plt.savefig(model_dir / 'policies-executions' / f'policy_{inp}.png')
