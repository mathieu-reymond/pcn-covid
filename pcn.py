import heapq
import numpy as np
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from pygmo import hypervolume
from logger import Logger
import wandb
import pickle


def crowding_distance(points, ranks=None):
    crowding = np.zeros(points.shape)
    # compute crowding distance separately for each non-dominated rank
    if ranks is None:
        ranks = non_dominated_rank(points)
    unique_ranks = np.unique(ranks)
    for rank in unique_ranks:
        current_i = ranks == rank
        current = points[current_i]
        if len(current) == 1:
            crowding[current_i] = 1
            continue
        # first normalize accross dimensions
        current = (current-current.min(axis=0))/(current.ptp(axis=0)+1e-8)
        # sort points per dimension
        dim_sorted = np.argsort(current, axis=0)
        point_sorted = np.take_along_axis(current, dim_sorted, axis=0)
        # compute distances between lower and higher point
        distances = np.abs(point_sorted[:-2] - point_sorted[2:])
        # pad extrema's with 1, for each dimension
        distances = np.pad(distances, ((1,), (0,)), constant_values=1)
        
        current_crowding = np.zeros(current.shape)
        current_crowding[dim_sorted, np.arange(points.shape[-1])] = distances
        crowding[current_i] = current_crowding
    # sum distances of each dimension of the same point
    crowding = np.sum(crowding, axis=-1)
    # normalized by dividing by number of objectives
    crowding = crowding/points.shape[-1]
    return crowding


def non_dominated_rank(points):
    ranks = np.zeros(len(points), dtype=np.float32)
    current_rank = 0
    # get unique points to determine their non-dominated rank
    unique_points, indexes = np.unique(points, return_inverse=True, axis=0)
    # as long as we haven't processed all points
    while not np.all(unique_points==-np.inf):
        _, nd_i = non_dominated(unique_points, return_indexes=True)
        # use indexes to compute inverse of unique_points, but use nd_i instead
        ranks[nd_i[indexes]] = current_rank
        # replace ranked points with -inf, so that they won't be non-dominated again
        unique_points[nd_i] = -np.inf
        current_rank += 1
    return ranks


def epsilon_metric(coverage_set, pareto_front):
    # normalize pareto front and coverage set for each 
    min_, ptp = pareto_front.min(axis=0),pareto_front.ptp(axis=0)
    pareto_front = (pareto_front-min_)/(ptp+1e-8)
    coverage_set = (coverage_set-min_)/(ptp+1e-8)
    # for every point in the pareto front, find the closest point in the coverage set
    # do this for every dimension separately
    # duplicate every point of the PF to compare with every point of the CS
    pf_duplicate = np.tile(np.expand_dims(pareto_front, 1), (1, len(coverage_set), 1))
    # distance for each dimension, for each point
    epsilon = np.abs(pf_duplicate-coverage_set)
    # for each point, take the maximum epsilon with pareto front
    epsilon = epsilon.max(-1)
    # closest point (in terms of epsilon) with PF
    epsilon = epsilon.min(-1)

    return epsilon


@dataclass
class Transition(object):
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    terminal: bool

device = 'cpu'

def non_dominated(solutions, return_indexes=False):
    is_efficient = np.ones(solutions.shape[0], dtype=bool)
    for i, c in enumerate(solutions):
        if is_efficient[i]:
            # Remove dominated points, will also remove itself
            is_efficient[is_efficient] = np.any(solutions[is_efficient] > c, axis=1)
            # keep this solution as non-dominated
            is_efficient[i] = 1
    if return_indexes:
        return solutions[is_efficient], is_efficient
    else:
        return solutions[is_efficient]


def compute_hypervolume(q_set, ref):
    nA = len(q_set)
    q_values = np.zeros(nA)
    for i in range(nA):
        # pygmo uses hv minimization,
        # negate rewards to get costs
        points = np.array(q_set[i]) * -1.
        hv = hypervolume(points)
        # use negative ref-point for minimization
        q_values[i] = hv.compute(ref*-1)
    return q_values

def nlargest(n, experience_replay, objectives, threshold=.2):
    returns = np.array([e[2][0].reward for e in experience_replay])
    # keep only used objectives
    returns = returns[:, objectives]
    # crowding distance of each point, check ones that are too close together
    distances = crowding_distance(returns)
    sma = np.argwhere(distances <= threshold).flatten()

    nd, nd_i = non_dominated(returns, return_indexes=True)
    nd = returns[nd_i]
    # we will compute distance of each point with each non-dominated point,
    # duplicate each point with number of nd to compute respective distance
    returns_exp = np.tile(np.expand_dims(returns, 1), (1, len(nd), 1))
    # distance to closest nd point
    l2 = np.min(np.linalg.norm(returns_exp-nd, axis=-1), axis=-1)*-1
    # all points that are too close together (crowding distance < threshold) get a penalty
    nd_i = np.nonzero(nd_i)[0]
    _, unique_i = np.unique(nd, axis=0, return_index=True)
    unique_i = nd_i[unique_i]
    duplicates = np.ones(len(l2), dtype=bool)
    duplicates[unique_i] = False
    l2[duplicates] -= 1e-5
    l2[sma] -= 1

    sorted_i = np.argsort(l2)
    largest = [experience_replay[i] for i in sorted_i[-n:]]
    # before returning largest elements, update all distances in heap
    for i in range(len(l2)):
        experience_replay[i] = (l2[i], experience_replay[i][1], experience_replay[i][2])
    heapq.heapify(experience_replay)
    return largest

def add_episode(transitions, experience_replay, gamma=1., max_size=100, step=0):
    # compute return
    for i in reversed(range(len(transitions)-1)):
        transitions[i].reward += gamma * transitions[i+1].reward
    # pop smallest episode of heap if full, add new episode
    # heap is sorted by negative distance, (updated in nlargest)
    # put positive number to ensure that new item stays in the heap
    if len(experience_replay) == max_size:
        heapq.heappushpop(experience_replay, (1, step, transitions))
    else:
        heapq.heappush(experience_replay, (1, step, transitions))

def choose_action(model, obs, desired_return, desired_horizon, eval=False):
    # if observation is not a simple np.array, convert individual arrays to tensors
    obs = [torch.tensor([o]).to(device) for o in obs] if type(obs) == tuple else torch.tensor([obs]).to(device)
    log_probs = model(obs,
                      torch.tensor([desired_return]).to(device),
                      torch.tensor([desired_horizon]).unsqueeze(1).to(device))
    log_probs = log_probs.detach().cpu().numpy()[0]
    # check if actions are continuous
    # TODO hacky
    if model.__class__.__name__ == 'ContinuousHead':
        action = log_probs
        # add some noise for randomness
        if not eval:
            action = np.clip(action + np.random.normal(0, 0.1, size=action.shape).astype(np.float32), 0, 1)
    else:
        # if evaluating: act greedily
        if eval:
            return np.argmax(log_probs, axis=-1)
        if log_probs.ndim == 1:
            action = np.random.choice(np.arange(len(log_probs)), p=np.exp(log_probs))
        elif log_probs.ndim == 2:
            action = np.array(list([np.random.choice(np.arange(len(lp)), p=np.exp(lp)) for lp in log_probs]))
    return action

def run_episode(env, model, desired_return, desired_horizon, max_return, eval=False):
    transitions = []
    obs = env.reset()
    done = False
    while not done:
        action = choose_action(model, obs, desired_return, desired_horizon, eval=eval)
        n_obs, reward, done, _ = env.step(action)

        transitions.append(Transition(
            observation=obs,
            action=action,
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

def choose_commands(experience_replay, n_episodes, objectives, threshold=0.2):
    # get best episodes, according to their crowding distance
    episodes = nlargest(n_episodes, experience_replay, objectives ,threshold=threshold)
    returns, horizons = list(zip(*[(e[2][0].reward, len(e[2])) for e in episodes]))
    # keep only non-dominated returns
    returns, nd_i = non_dominated(np.array(returns), return_indexes=True)
    horizons = np.array(horizons)[nd_i]
    # pick random return from random best episode
    r_i = np.random.randint(0, len(returns))
    desired_horizon = np.float32(horizons[r_i]-2)
    # mean and std per objective
    m, s = np.mean(returns, axis=0), np.std(returns, axis=0)
    # desired return is sampled from [M, M+S], to try to do better than mean return
    desired_return = returns[r_i].copy()
    # random objective
    r_i = np.random.choice(objectives)
    desired_return[r_i] += np.random.uniform(high=s[r_i])
    desired_return = np.float32(desired_return)
    return desired_return, desired_horizon

def update_model(model, opt, experience_replay, batch_size, noise=0.):
    batch = []
    # randomly choose episodes from experience buffer
    s_i = np.random.choice(np.arange(len(experience_replay)), size=batch_size, replace=True)
    for i in s_i:
        # episode is tuple (return, transitions)
        ep = experience_replay[i][2]
        # choose random timestep from episode, 
        # use it's return and leftover timesteps as desired return and horizon
        t = np.random.randint(0, len(ep))
        # reward contains return until end of episode
        s_t, a_t, r_t, h_t = ep[t].observation, ep[t].action, np.float32(ep[t].reward), np.float32(len(ep)-t)
        batch.append((s_t, a_t, r_t, h_t))

    obs, actions, desired_return, desired_horizon = zip(*batch)
    # since each state is a tuple with (compartment, events, prev_action), reorder obs
    obs = zip(*obs)
    obs = tuple([torch.tensor(o).to(device) for o in obs])
    # TODO TEST add noise to the desired return
    desired_return = torch.tensor(desired_return).to(device)
    desired_return = desired_return + noise*torch.normal(0, 1, size=desired_return.shape, device=desired_return.device)
    log_prob = model(obs,
                     desired_return,
                     torch.tensor(desired_horizon).unsqueeze(1).to(device))

    opt.zero_grad()
    # check if actions are continuous
    # TODO hacky
    if model.__class__.__name__ == 'ContinuousHead':
        l = F.mse_loss(log_prob, torch.tensor(actions))
    else:
        # one-hot of action for CE loss
        actions = torch.tensor(actions).long().to(device)
        actions = F.one_hot(actions, num_classes=log_prob.shape[-1])
        # cross-entropy loss
        l = torch.sum(-actions*log_prob, -1).sum(-1)
    l = l.mean()
    l.backward()
    opt.step()

    return l, log_prob


def eval(env, model, coverage_set, horizons, max_return, gamma=1., n=10):
    e_returns = np.empty((coverage_set.shape[0], n, coverage_set.shape[-1]))
    all_transitions = []
    for e_i, target_return, horizon in zip(np.arange(len(coverage_set)), coverage_set, horizons):
        n_transitions = []
        for n_i in range(n):
            transitions = run_episode(env, model, target_return, np.float32(horizon), max_return, eval=True)
            # compute return
            for i in reversed(range(len(transitions)-1)):
                transitions[i].reward += gamma * transitions[i+1].reward
            e_returns[e_i, n_i] = transitions[0].reward
            n_transitions.append(transitions)
        all_transitions.append(n_transitions)

    return e_returns, all_transitions


def train(env, 
          model,
          learning_rate=1e-2,
          batch_size=1024, 
          total_steps=1e7,
          n_model_updates=100,
          n_step_episodes=10,
          n_er_episodes=500,
          gamma=1.,
          max_return=250.,
          max_size=500,
          ref_point=np.array([0, 0]),
          threshold=0.2,
          noise=0.0,
          objectives=None,
          n_evaluations=10,
          logdir='runs/'):
    step = 0
    if objectives == None:
        objectives = tuple([i for i in range(len(ref_point))])
    total_episodes = n_er_episodes
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    logger = Logger(logdir=logdir)
    n_checkpoints = 0
    # fill buffer with random episodes
    experience_replay = []
    for _ in range(n_er_episodes):
        transitions = []
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            n_obs, reward, done, _ = env.step(action)
            transitions.append(Transition(obs, action, np.float32(reward).copy(), n_obs, done))
            obs = n_obs
            step += 1
        # add episode in-place
        add_episode(transitions, experience_replay, gamma=gamma, max_size=max_size, step=step)
    while step < total_steps:
        loss = []
        entropy = []
        for _ in range(n_model_updates):
            l, lp = update_model(model, opt, experience_replay, batch_size=batch_size, noise=noise)
            loss.append(l.detach().cpu().numpy())
            lp = lp.detach().cpu().numpy()
            ent = np.sum(-np.exp(lp)*lp)
            entropy.append(ent)

        desired_return, desired_horizon = choose_commands(experience_replay, n_er_episodes, objectives)

         # get all leaves, contain biggest elements, experience_replay got heapified in choose_commands
        leaves = np.array([(len(e[2]), e[2][0].reward) for e in experience_replay[len(experience_replay)//2:]])
        e_lengths, e_returns = zip(*leaves)
        e_lengths, e_returns = np.array(e_lengths), np.array(e_returns)
        try:
            if len(experience_replay) == max_size:
                logger.put('train/leaves', e_returns, step, f'{e_returns.shape[-1]}d')
            # hv = hypervolume(e_returns[...,objectives]*-1)
            # hv_est = hv.compute(ref_point[objectives]*-1)
            # logger.put('train/hypervolume', hv_est, step, 'scalar')
            # wandb.log({'hypervolume': hv_est}, step=step)
        except ValueError:
            pass

        returns = []
        horizons = []
        for _ in range(n_step_episodes):
            transitions = run_episode(env, model, desired_return, desired_horizon, max_return)
            step += len(transitions)
            add_episode(transitions, experience_replay, gamma=gamma, max_size=max_size, step=step)
            returns.append(transitions[0].reward)
            horizons.append(len(transitions))
        
        total_episodes += n_step_episodes
        logger.put('train/episode', total_episodes, step, 'scalar')
        logger.put('train/loss', np.mean(loss), step, 'scalar')
        logger.put('train/entropy', np.mean(entropy), step, 'scalar')
        logger.put('train/horizon/desired', desired_horizon, step, 'scalar')
        logger.put('train/horizon/distance', np.linalg.norm(np.mean(horizons)-desired_horizon), step, 'scalar')
        for o in range(len(desired_return)):
            logger.put(f'train/return/{o}/value', desired_horizon, step, 'scalar')
            logger.put(f'train/return/{o}/desired', np.mean(np.array(returns)[:, o]), step, 'scalar')
            logger.put(f'train/return/{o}/distance', np.linalg.norm(np.mean(np.array(returns)[:, o])-desired_return[o]), step, 'scalar')
        print(f'step {step} \t return {np.mean(returns, axis=0)}, ({np.std(returns, axis=0)}) \t loss {np.mean(loss):.3E}')
        
        # compute hypervolume of leaves
        valid_e_returns = e_returns[np.all(e_returns[:,objectives] >= ref_point[objectives,], axis=1)]
        hv = compute_hypervolume(np.expand_dims(valid_e_returns[:,objectives], 0), ref_point[objectives,])[0] if len(valid_e_returns) else 0

        wandb.log({
            'episode': total_episodes,
            'episode_steps': np.mean(horizons),
            'loss': np.mean(loss),
            'entropy': np.mean(entropy),
            'hypervolume': hv,
        }, step=step)

        if step >= (n_checkpoints+1)*total_steps/10:
            torch.save(model, f'{logger.logdir}/model_{n_checkpoints+1}.pt')
            n_checkpoints += 1

            coverage_set_table = wandb.Table(data=e_returns, columns=[f'o_{o}' for o in range(e_returns.shape[1])])

            # current coverage set
            _, e_i = non_dominated(e_returns[:,objectives], return_indexes=True)
            e_returns = e_returns[e_i]
            e_lengths = e_lengths[e_i]

            e_r, t_r = eval(env, model, e_returns, e_lengths, max_return, gamma=gamma, n=n_evaluations)
            # save raw evaluation returns
            logger.put(f'eval/returns/{n_checkpoints}', e_r, 0, f'{len(e_r)}d')
            # compute e-metric
            epsilon = epsilon_metric(e_r[...,objectives].mean(axis=1), e_returns[...,objectives])
            logger.put('eval/epsilon/max', epsilon.max(), step, 'scalar')
            logger.put('eval/epsilon/mean', epsilon.mean(), step, 'scalar')
            print('='*10, ' evaluation ', '='*10)
            for d, r in zip(e_returns, e_r):
                print('desired: ', d, '\t', 'return: ', r.mean(0))
            print(f'epsilon max/mean: {epsilon.max():.3f} \t {epsilon.mean():.3f}')
            print('='*22)

            nd_coverage_set_table = wandb.Table(data=e_returns*env.scale[None], columns=[f'o_{o}' for o in range(e_returns.shape[1])])
            nd_executions_table = wandb.Table(data=e_r.mean(axis=1)*env.scale[None], columns=[f'o_{o}' for o in range(e_returns.shape[1])])
            
            executions_transitions = wandb.Artifact(
                f'run-{wandb.run.id}-execution-transitions', type='transitions'
            )
            with executions_transitions.new_file('transitions.pkl', 'wb') as f:
                pickle.dump(t_r, f)

            wandb.log({
                'coverage_set': coverage_set_table,
                'nd_coverage_set': nd_coverage_set_table,
                'executions': nd_executions_table,
                'eps_max': epsilon.max(),
                'eps_mean': epsilon.mean(),
            }, step=step)
            wandb.run.log_artifact(executions_transitions)
