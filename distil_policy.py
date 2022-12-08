import pickle
import pathlib

from numpy.random import beta
import torch
from torch.utils.data import TensorDataset, DataLoader
from dataclasses import dataclass
from SoftDecisionTree.sdt.model import SoftDecisionTree
import wandb
import os
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# with open('/tmp/tables.pkl', 'rb') as f:
#     tables = pickle.load(f)
# with open('/tmp/configs.pkl', 'rb') as f:
#     configs = pickle.load(f)

batch_size = 32
epochs = 2000

@dataclass
class Transition(object):
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    terminal: bool

@dataclass
class AugmentedTransition(object):
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    terminal: bool
    t_r: np.ndarray
    t_h: int


@dataclass
class SDTARGS(object):
    depth : int
    lmbda : float
    input_dim : int
    output_dim: int
    log_interval: int
    save : str
    tensorboard : bool
    device: str


def download_policy_logs(run_id):
    path = f'mreymond/pcn-covid/runs/{run_id}'
    api = wandb.Api(timeout=30)
    run = api.run(path=path)
    # get artifacts from run
    run_dict = {}
    for name in ('coverage_set', 'nd_coverage_set', 'executions', 'execution-transitions'):
        artifact_path = f'mreymond/{run.project}/run-{run.id}-{name}:latest'
        print(f'getting: {artifact_path}')
        # download artifact
        try:
            file_path = api.artifact(artifact_path).file(f'/tmp')
            if file_path[-4:] == '.pkl':
                # load the downloaded file
                with open(file_path, 'rb') as f:
                    table = pickle.load(f)
            else:
                # load the downloaded file
                with open(file_path, 'r') as f:
                    table = json.load(f)
            os.remove(file_path)
            run_dict[name] = table
            time.sleep(3)
        except Exception as e:
            print(f'skipping... {e}')
            run_dict[name] = None
    config = run.config
    config['id'] = run.id
    config['name'] = run.name
    return run_dict, config


def train_sdt(transitions, objectives=[1,5], with_targets=False, policy_specific=False):
    if policy_specific:
        assert not with_targets, 'with targets only for multi-policy SDT'
    # take one of the policies, by index
    # policy_runs = transitions[policy_i]
    # create dataset
    x, y, sdt = [], [], []
    # use all learnt policies
    for p_i, policy_runs in enumerate(transitions):
        policy_i = np.zeros(len(transitions))
        policy_i[p_i] = 1
        for run in policy_runs:
            for i, transition in enumerate(run):
                # compartments and school holidays
                x_t = np.concatenate((transition.observation[0].flatten(), transition.observation[1]))
                # append target return, target horizon
                if with_targets:
                    x_t = np.concatenate((x_t, policy_i))
                    # x_t = np.concatenate((x_t, transition.reward.flatten()[objectives], [len(run)-i]))
                x.append(torch.from_numpy(x_t).float())
                y.append(torch.from_numpy(transition.action).float())

            if policy_specific:
                sdt_ = _train_sdt(x, y, policy_specific)
                sdt.append(sdt_)
                x, y = [], []

    if not policy_specific:
        sdt = _train_sdt(x, y, policy_specific)

    return sdt


def _train_sdt(x, y, policy_specific):
    x = torch.stack(x, 0).reshape(len(x), -1)
    y = torch.stack(y, 0)
    # normalize x-data
    x_max = torch.max(x, dim=0, keepdims=True).values + 1e-5
    x = x/x_max
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    depth = 2 if policy_specific else 6
    
    args = SDTARGS(
        depth,
        0.1,
        x_max.shape[-1],
        3,
        20,
        '/tmp/sdt_results/',
        False,
        'cpu'
    )

    sdt = SoftDecisionTree(args, normalization=x_max)
    for epoch in range(epochs):
        sdt.train_(dataloader, epoch)
    return sdt


def visualize_sdt(sdt):
    weights = sdt.weights.detach().numpy()
    betas = sdt.betas.detach().numpy()
    biases = sdt.biases.detach().numpy()

    # probs_right = torch.sigmoid(sdt.betas * torch.addmm(sdt.biases, x, sdt.weights.t()))
    # probs_right = probs_right.detach().numpy()
    # probs_left = 1-probs_right
    depths = sdt.depth
    fig, axs = plt.subplots(depths, 2**depths)
    for depth in range(depths):
        for i in range(2**depth):
            ax = axs[depth, i]
            w = weights[2**depth+i-1]
            compartments = w[:130].reshape((10, 13))
            school_holidays = w[130:131]
            t_r = w[131:133]
            t_h = w[133:]
            ax.imshow(compartments)
    plt.show()


def aggregate_compartments(compartments):
    # (10, 13) matrix with age-groups and compartments
    # 0-10, 10-20
    youth = np.mean(compartments[:2], axis=0)
    # 20-30, 30-40, 40-50, 50-60
    adults = np.mean(compartments[2:6], axis=0)
    # 60+
    elderly = np.mean(compartments[6:], axis=0)
    yae = np.stack([youth, adults, elderly], axis=0)
    # I compartments 'I_presym', 'I_asym', 'I_mild', 'I_sev',
    i_ = np.mean(yae[:,2:6], axis=1)
    # H compartments
    h_ = np.mean(yae[:,6:8], axis=1)
    # H_new compartments
    n_ = np.mean(yae[:,10:12], axis=1)
    compartments = np.stack([yae[:,0], yae[:,1], i_, h_, n_], axis=1)
    return compartments


def visualize_sdt2(sdt, transitions=None, policy_i=None, with_targets=False, with_aggregate_compartments=False):
    weights = sdt.weights.detach().numpy()
    biases = sdt.biases.detach().numpy()
    betas = sdt.betas.detach().numpy()
    depths = sdt.depth

    # compute positions of decision tree nodes
    specs = []
    for depth in range(depths, 0, -1):
        start = 2**(depth-1)-1
        step = 2**depth
        spots = [p for p in range(start, 2**depths, step)]
        spec = [{'colspan': 2} if i in spots else None for i in range(2**depths)]
        specs.append(spec)
    specs.append([{'colspan': 2**(depths-1), 'secondary_y': True} if i == 0 else None for i in range(2**depths)])

    fig = make_subplots(
        rows=depths+1, cols=2**depths,
        specs=deepcopy(specs),
        print_grid=False)

    specs_ = deepcopy(specs)
    for depth in range(depths):
        for i in range(2**depth):
            w = weights[2**depth+i-1]
            compartments = w[:130].reshape((10, 13))
            if with_aggregate_compartments:
                compartments = aggregate_compartments(compartments)
            school_holidays = w[130:131]
            extras = np.full(compartments.shape[0], np.nan)
            extras[0] = school_holidays
            # if with_targets:
            #     t_r = w[131:133]
            #     t_h = w[133:]
            #     extras[5:7] = t_r
            #     extras[9] = t_h
            compartments = np.concatenate((compartments, extras[:, None]), 1)
            # get figure position
            col = specs_[depth].index({'colspan': 2})
            # used this position, remove from speclist
            specs_[depth][col] = None
            fig.add_trace(go.Heatmap(z=compartments, legendgroup='1'), row=depth+1, col=col+1)
    fig.update_traces(showscale=False)

    if transitions is not None:
        # =================
        # INTERACTIVITY
        # =================
        # create separate heatmaps for each transition
        timestep_plots = [np.arange(len(fig.data))]
        for t_i, transition in enumerate(transitions):
            specs_ = deepcopy(specs)
            nb_plots = len(fig.data)

            states = np.array([t.observation[0] for t in transitions])
            # add final state
            states = np.concatenate((states, transitions[-1].next_observation[0][None]), axis=0)
            ari = (states[:-1,:,0]-states[1:,:,0]).sum(axis=-1)
            i_hosp_new = states[...,-3].sum(axis=-1)
            i_icu_new = states[...,-2].sum(axis=-1)
            d_new = states[...,-1].sum(axis=-1)
            actions = np.array([t.action for t in transitions])
            # append action of None
            actions = np.concatenate((actions, [[None]*3]))
            
            timesteps = np.arange(len(states))
            # make plots for hospitalizations
            fig.add_trace(go.Scatter(x=timesteps, y=i_hosp_new+i_icu_new, name='hosp_tot', mode='lines', legendgroup='2'), secondary_y=False, row=len(specs), col=1)
            fig.add_trace(go.Scatter(x=timesteps, y=i_hosp_new, name='hosp', mode='lines', legendgroup='2'), secondary_y=False, row=len(specs), col=1)
            fig.add_trace(go.Scatter(x=timesteps, y=i_icu_new, name='icu', mode='lines', legendgroup='2'), secondary_y=False, row=len(specs), col=1)
            fig.add_trace(go.Scatter(x=timesteps, y=d_new, name='deaths', mode='lines', legendgroup='2'), secondary_y=False, row=len(specs), col=1)
            fig.add_trace(go.Scatter(x=[t_i, t_i], y=[0, 1], mode='lines'), secondary_y=True, row=len(specs), col=1)
            # make plots for actions
            for a_i, a in enumerate(('p_w', 'p_s', 'p_l')):
                fig.add_trace(
                    go.Scatter(x=timesteps, y=actions[:, a_i], name=a, mode='lines', line={'dash': 'dash'}, legendgroup='3'),
                    # layout_yaxis_range=[-0.1,1.1],
                    secondary_y=True, row=len(specs), col=1)
            fig.update_yaxes(range=[-0.1,1.1], secondary_y=True)

            # compute weights for each compartment variable
            obs = transition.observation
            obs = np.concatenate((obs[0].flatten(), obs[1].flatten()))
            if with_targets:
                if policy_i is not None:
                    obs = np.concatenate((obs, policy_i))
                else:
                    obs = np.concatenate((obs, transition.t_r.flatten()))
                    obs = np.concatenate((obs, transition.t_h))
            inp = (obs/sdt.normalization.detach().numpy())
            # multiply with weights before sigmoid
            w_inp = betas[:, None]*((inp*weights)+biases[:,None]/len(obs))
            probs_right = torch.sigmoid(torch.from_numpy(w_inp.sum(1))).numpy()
            for depth in range(depths):
                for i in range(2**depth):
                    w = w_inp[2**depth+i-1]
                    prob_right = probs_right[2**depth+i-1]
                    compartments = w[:130].reshape((10, 13))
                    if with_aggregate_compartments:
                        compartments = aggregate_compartments(compartments)
                    school_holidays = w[130:131]
                    extras = np.full(compartments.shape[0], np.nan)
                    extras[0] = school_holidays
                    # if with_targets:
                    #     w_t_r = w[131:133]
                    #     w_t_h = w[133:]
                    #     extras[5:7] = w_t_r
                    #     extras[9] = w_t_h
                    compartments = np.concatenate((compartments, extras[:, None]), 1)
                    # get figure position
                    col = specs_[depth].index({'colspan': 2})
                    # used this position, remove from speclist
                    specs_[depth][col] = None
                    fig.add_trace(go.Heatmap(z=compartments, legendgroup='1', visible=False), row=depth+1, col=col+1)
                    # prob color as mix between left (green) and right (red)
                    prob_color = np.array([255, 0, 0])*prob_right+ np.array([0,255,0])*(1-prob_right)
                    prob_color =  f'rgb({prob_color[0]:.0f}, {prob_color[1]:.0f}, {prob_color[2]:.0f})'
                    # add color rectangle in heatmap for prob
                    w, h = compartments.shape
                    fig.add_trace(go.Scatter(x=[0, h-1, h-1 , 0, 0], y=[0, 0, w-1, w-1, 0], mode='lines', line={'color': prob_color}, visible=False), row=depth+1, col=col+1)
            # fig.update_traces(showscale=False)
            # all these new plots are part of a separate slider step
            timestep_plots.append(np.arange(nb_plots, len(fig.data)))
        # create slider steps
        slider_steps = []
        for timestep in timestep_plots:
            step = dict(
                method='update',
                args=[{'visible': [i in timestep for i in range(len(fig.data))]}]
            )
            slider_steps.append(step)
        # create slider itself
        slider = {'active': 0, 'steps': slider_steps}
        fig.update_layout(sliders=[slider])

    fig.update_layout(title_text='decision tree', legend_tracegroupgap = 180)
    fig.show()


def predict(sdt, observation, desired_return, desired_horizon, policy_i=None, with_targets=False):
    compartments = observation[0].flatten()
    school_holidays = observation[1].flatten()
    observation = np.concatenate((compartments, school_holidays))
    if with_targets:
        if policy_i is not None:
            observation = np.concatenate((observation, policy_i))
        else:
            observation = np.concatenate((observation, desired_return, desired_horizon))
    observation = observation.reshape(1, -1).astype(np.float32)
    probs_right = torch.sigmoid(sdt.betas * torch.addmm(sdt.biases, observation/sdt.normalization, sdt.weights.t()))
    leaf_path_probs = sdt._calc_path_probs(probs_right, sdt.ancestors_leafs)

    leafs_pred = torch.sigmoid(sdt.leafs)
    # weighted output of decision tree:
    weighted_leaves_pred = leafs_pred[None]*leaf_path_probs[...,None]
    # sum over all leaves
    weighted_leaves_pred = weighted_leaves_pred.sum(dim=1)
    return weighted_leaves_pred.detach().numpy().flatten()


def execute_sdt(sdt, env, desired_return, desired_horizon, max_return, gamma=1, objectives=[1,5], policy_i=None, with_targets=False):
    transitions = []
    obs = env.reset()
    done = False
    while not done:
        action = predict(sdt, obs, desired_return, desired_horizon, policy_i=policy_i, with_targets=with_targets)
        n_obs, reward, done, _ = env.step(action)

        transitions.append(AugmentedTransition(
            observation=obs,
            action=env.action(action),
            reward=np.float32(reward).copy(),
            next_observation=n_obs,
            terminal=done,
            t_r=desired_return,
            t_h=desired_horizon
        ))

        obs = n_obs
        # clip desired return, to return-upper-bound, 
        # to avoid negative returns giving impossible desired returns
        # reward = np.array((reward[1], reward[2]))
        desired_return = np.clip(desired_return-reward[objectives], None, max_return, dtype=np.float32)
        # clip desired horizon to avoid negative horizons
        desired_horizon = np.float32(max(desired_horizon-1, 1.))

    # compute return
    for i in reversed(range(len(transitions)-1)):
        transitions[i].reward += gamma * transitions[i+1].reward
    return transitions


def n_execute_sdt(sdt, env, desired_return, desired_horizon, max_return, gamma=1, n=30, policy_i=None, with_targets=False):
    returns = np.empty((n, len(scale)))
    all_transitions = []
    for i in range(n):
        transitions = execute_sdt(sdt, env, desired_return, desired_horizon, max_return, gamma=1, policy_i=policy_i, with_targets=with_targets)
        returns[i] = transitions[0].reward
        all_transitions.append(transitions)
    return all_transitions, returns


def make_env(config, scale):
    env_type = 'ODE' if config['env'] == 'ode' else 'Binomial'

    ref_point = np.array([-15000000, -200000, -1000.0, -1000.0, -1000.0, -1000.0])/scale
    scaling_factor = torch.tensor([[1, 1, 1, 1, 1, 1, 0.1]])
    envs = {}
    for e_type in ('ODE', 'Binomial'):
        env = gym.make(f'BECovidWithLockdown{e_type}Continuous-v0')
        nA = np.prod(env.action_space.shape)
        env.action = lambda x: x
        env = TodayWrapper(env)
        env = ScaleRewardEnv(env, scale=scale)
        envs[e_type] = env
        print(env)

    env = envs[env_type]
    return env


if __name__ == '__main__':
    import sys
    import gym
    import gym_covid
    from main_pcn import CovidModel, MultiDiscreteHead, DiscreteHead, ContinuousHead, ScaleRewardEnv, TodayWrapper, multidiscrete_env
    from eval_pcn import plot_episode
    import pathlib
    import argparse

    parser = argparse.ArgumentParser(description='SDT')
    parser.add_argument('--id', default=None, type=str, help='wandb id of run to train on')
    parser.add_argument('--checkpoint', default=None, type=str, help='checkpoint file')
    parser.add_argument('--with-targets', action='store_true',
        help='When using a single, multi-policy SDT, add the policy-index as one-hot in the state-space')
    parser.add_argument('--policy-specific', action='store_true',
        help='make a specific policy per SDT, instead of making a global SDT that encompasses all policies')
    parser.add_argument('--aggregate-compartments', action='store_true',
        help='aggregate the compartment model to [youth, adults, elderly] and [S, E, I, H, H_new]')
    args = parser.parse_args()
    print(args)

    run_id = args.id
    with_targets = args.with_targets
    policy_specific = args.policy_specific
    if args.checkpoint is not None:
        print('loading checkpoint...')
        checkpoint = torch.load(args.checkpoint)
        config = checkpoint['config']
        transitions = checkpoint['transitions']
        desired_returns = checkpoint['desired_returns']
        sdt = checkpoint['sdt']
    else:
        print('downloading run and training SDT...')
        table, config = download_policy_logs(run_id)

        # model_dir = pathlib.Path(f'/tmp/{config["env"]}_{config["id"]}')
        # model_dir.mkdir(parents=True, exist_ok=True)
        transitions = table['execution-transitions']

        penalty_term = 1e6
        scale = np.array([800000, 10000, 50., 20, 50, 90])
        def sort_without_penalty(t):
            arh = t[0][0].reward[1]*scale[1]
            if arh < -penalty_term:
                arh += penalty_term
            return arh
        transitions = sorted(transitions, key=sort_without_penalty)
        desired_returns = np.array([t[0][0].reward for t in transitions])
        print(desired_returns)

        sdt = train_sdt(transitions, with_targets=with_targets, policy_specific=policy_specific)
        print('saving checkpoint')
        checkpoint = {
            'config': config,
            'transitions': transitions,
            'desired_returns': desired_returns,
            'sdt': sdt
        }
        torch.save(checkpoint, f'SoftDecisionTree/saves/sdt_{run_id}_wt-{with_targets}_ps-{policy_specific}.pt')

    objectives = [1,5]
    scale = np.array([800000, 10000, 50., 20, 50, 90])
    max_return = np.array([0, 0, 0, 0, 0, 0])/scale
    max_return = max_return[objectives]
    env = make_env(config, scale)

    inp = -1
    while True:
        print('solutions: ')
        for i, p in enumerate(desired_returns):
            print(f'{i} : {p[objectives]}')
        inp = input('-> ')
        p_i = int(inp)

        desired_return = desired_returns[p_i][objectives]
        desired_horizon = np.array([17.], dtype=np.float32)
        policy_i = np.zeros(len(desired_returns))
        policy_i[p_i] = 1
        if policy_specific:
            policy = sdt[p_i]
        else:
            policy = sdt

        all_transitions, returns = n_execute_sdt(policy, env, desired_return, desired_horizon, max_return, n=1, policy_i=policy_i, with_targets=with_targets)
        returns = returns*scale[None]

        print(f'target: {desired_return*scale[objectives]} \t obtained {returns[:,objectives]}')
        # print(np.mean(returns, 0))

        # plt.subplots(3, 1, sharex=True)
        # for transitions in all_transitions:
        #     plot_episode(transitions, 0.1)
        # plt.show()
        # plt.close()

        visualize_sdt2(policy, all_transitions[0], policy_i=policy_i, with_targets=with_targets, with_aggregate_compartments=args.aggregate_compartments)
