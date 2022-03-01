from pathlib import Path
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import interp1d
from metrics import compute_hypervolume, non_dominated, epsilon_metric
import sys
import glob
import pandas as pd

pre = sys.argv[1]


def udrl_runs(logdir):
    runs = []
    for i, path in enumerate(Path(logdir).rglob('log.h5')):
        print('making plots for ' + str(path))
        with h5py.File(path, 'r') as logfile:
            pf = logfile['train/leaves/ndarray']
            try:
                hv = logfile['train/hypervolume']
            except KeyError:
                # recompute hypervolume
                ref_points = {
                    'dst':np.array([0, -200.]),
                    'walkroom2':np.array([-20.0]*2),
                    'walkroom3':np.array([-20.0]*3),
                    'walkroom4':np.array([-20.0]*4),
                    'walkroom5':np.array([-20.0]*5),
                    'walkroom6':np.array([-10.0]*6),
                    'walkroom7':np.array([-10.0]*7),
                    'walkroom8':np.array([-10.0]*8),
                    'walkroom9':np.array([-10.0]*9),
                    'ode':np.array([-50000, -2000.0])/np.array([10000, 100.]),
                    'binomial':np.array([-50000, -2000.0])/np.array([10000, 100.])
                }
                for rp in ref_points.keys():
                    if rp in logdir:
                        ref_point = ref_points[rp]
                # only keep points that dominate ref point
                hv = []
                for f in pf:
                    valid_points = f[np.all(f >= ref_point, axis=1)]
                    if len(valid_points) == 0:
                        hv.append(0)
                    else:
                        hv.append(compute_hypervolume(valid_points, ref_point))
                steps = logfile['train/leaves/step']
                hv = np.stack((steps, hv), axis=1)
            # best points found over training
            bp = non_dominated(np.concatenate(pf, axis=0))
            run = {
                'pareto_front': pf[-1],
                'hypervolume': hv[:],
                'best_points': bp
            }
            runs.append(run)
    return runs


def show_run(run):
    pf = run['pareto_front']
    hv = run['hypervolume']
    plt.figure()
    plt.scatter(*[pf[:, i] for i in range(pf.shape[-1])])
    plt.show()
    plt.close()

    plt.figure()
    plt.plot(hv[:, 0], hv[:, 1])
    plt.show()
    plt.close()


def show_hypervolume(env):
    plt.figure()

    logdirs = env_logdirs[env]
    for k, v in logdirs.items():
        runs = algo_runs[k](v)
        hv = [r['hypervolume'][-1][1] for r in runs]
        plt.scatter([k]*len(hv), hv)
    plt.show()


def show_pareto_front(all_runs):
    plt.figure()
    if env == 'minecart':
        plt.gcf().add_subplot(111, projection='3d')
    cmaps = {
        'udrl': 'Blues',
        'mones': 'Oranges',
        'ra': 'Greens',
    }

    logdirs = env_logdirs[env]
    for k, v in logdirs.items():
        runs = algo_runs[k](v)
        for i, run in enumerate(runs):
            pf = run['pareto_front']
            weights = [i]*len(pf)
            plt.gca().scatter(*[pf[:,j] for j in range(pf.shape[1])], c=weights, vmin=-1, vmax=len(runs)+1, cmap=cmaps[k])
    plt.show()


def plot_pareto_front(all_runs, jitter=0.01):
    plt.figure()
    plt.title('pareto front')

    nO = list(all_runs.values())[0][0]['pareto_front'].shape[-1]
    if nO == 3:
        plt.gcf().add_subplot(111, projection='3d')
    elif nO > 3:
        print('aborting since more than 3 objectives')
        plt.close()
        return

    for k, v in all_runs.items():
        points = [run['pareto_front'] for run in v]
        selected = []
        for p in points:
            p = np.unique(p, axis=0)
            selected = p if len(p) >= len(selected) else selected
        p = selected
        print(p)
        jittered_p = p # + np.random.normal(0, p.std(axis=0, keepdims=True)*jitter, size=p.shape)
        coords = list(zip(*jittered_p))
        coords = np.array(coords)*np.array([[10000], [100]])
        plt.gca().scatter(*coords, alpha=0.2, label=f'{k}')
        plt.xlabel('total number of daily-new-hospitalizations')
        plt.ylabel('social burden as cumulative contacts lost per person')

        df = pd.DataFrame(data=p, columns=[f'o{i}' for i in range(nO)])
        df.to_csv(f'/tmp/{k}.csv', index=False)
    
    plt.legend()
    plt.show()


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'full')[:-w+1] / w

def interpolate_runs(runs, w=100):
    all_steps = np.array(sorted(np.unique(np.concatenate([r[:,0] for r in runs]))))
    all_values = np.stack([np.interp(all_steps, r[:,0], r[:,1]) for r in runs], axis=0)
    return all_steps, all_values


def plot_hypervolume(all_runs):
    plt.figure()
    plt.title('hypervolume')

    for k, v in all_runs.items():
        steps, values = interpolate_runs([run['hypervolume'] for run in v])

        avg, std = np.mean(values, axis=0), np.std(values, axis=0)
        plt.plot(steps, avg, label=f'{k}')
        plt.fill_between(steps, avg-std, avg+std, alpha=0.2)
    
    plt.legend()
    plt.show()


def common_hypervolume(all_runs):
    all_hp = {}
    smallest_step = np.inf
    to_keep = 100
    for k, v in all_runs.items():
        steps, values = interpolate_runs([run['hypervolume'] for run in v])
        offset, interval = len(steps)%to_keep-1, len(steps)//to_keep
        to_keep_i = np.zeros(len(steps), dtype=bool)
        to_keep_i[offset::interval] = True
        to_keep_values = values[:, to_keep_i]
        df = pd.DataFrame.from_dict({'steps':steps[to_keep_i]}|{f'run-{i}': to_keep_values[i] for i in range(len(values))})
        all_hp[k] = df

        # avg, std = np.mean(values, axis=0), np.std(values, axis=0)
        # all_hp[k] = (steps, avg, std)
        # smallest_step = steps[-1] if smallest_step > steps[-1] else smallest_step

    return all_hp
    # smallest_step = 2e5

    print('='*5 + f'Hypervolume ({smallest_step} steps)' + '='*5)
    for k, v in all_hp.items():
        limit = np.abs(v[0]-smallest_step).argmin()
        print(f'{k} : {v[1][limit]} +- {v[2][limit]}')


def non_dominated_across_runs(all_runs):
    all_bp = np.concatenate([np.concatenate([run['best_points'] for run in v]) for v in all_runs.values()])
    all_bp = non_dominated(all_bp)
    return all_bp


def compare_epsilon(all_runs, pareto_front):
    all_eps = {}
    for k, v in all_runs.items():
        eps = [epsilon_metric(run['pareto_front'], pareto_front) for run in v]
        eps = np.array(eps)
        all_eps[f'{k}-max'] = eps.max(axis=1)
        all_eps[f'{k}-mean'] = eps.mean(axis=1)
    # return all_eps
    df = pd.DataFrame.from_dict(all_eps)
    return df
    
    print('='*5 + f'Epsilon of final coverage set (max, mean)' + '='*5)
    for k, v in all_eps.items():
        print(f'{k} : {v[0]} +- {v[1]}')



if __name__ == '__main__':
    import argparse
    import warnings
    import gym

    print("""RECOMMENDED:
    DST:
     - MONES HV
     - RA 32
    MINECART
     - MONES HV
    SUMO
     - MONES ND
     - RA 0324
    """)

    parser = argparse.ArgumentParser(description='plots')
    parser.add_argument('--logs', required=True, type=str, nargs='+')
    parser.add_argument('--algo', required=True, type=str, nargs='+', help='udrl, mones or ra')
    parser.add_argument('--save-pf', type=str, default=None, 
        help='get all best points across all selected runs, and save them to file')
    args = parser.parse_args()

    assert len(args.algo) == len(args.logs), 'each log should refer to an algo'
    all_runs = {}
    for algo, logdir in zip(args.algo, args.logs):
        if algo.startswith('udrl'):
            get_runs = udrl_runs
        else:
            raise ValueError('unknown algo')
        runs = get_runs(logdir)

        all_runs[algo] = runs

    if args.save_pf is not None:
        try:
            with open(args.save_pf + '.npy', 'rb') as f:
                nd = np.load(f)
        except FileNotFoundError as e:
            if not 'walkroom' in args.save_pf:
                nd = non_dominated_across_runs(all_runs)
            else:
                nO = int(args.save_pf[-1:])
                env = gym.make(f'Walkroom{nO}D-v0')
                nd = env.pareto_front
            with open(args.save_pf + '.npy', 'wb') as f:
                np.save(f, nd)
    else:
        nd = non_dominated_across_runs(all_runs)
        warnings.warn('not using approximated pareto front')

    plot_pareto_front(all_runs)
    # hv = common_hypervolume(all_runs)
    plot_hypervolume(all_runs)


    #df = compare_epsilon(all_runs, nd)
    #eps_path = '/tmp/ra-' + Path(args.save_pf).name + '.csv'
    #df.to_csv(eps_path)

    #hv = common_hypervolume(all_runs)
    #for k, df in hv.items():
        #eps_path = '/tmp/hv-' + Path(args.save_pf).name + f'-{k}.csv'
        #df.to_csv(eps_path)

    """
    python experiment_plots.py --logs /tmp/udrl_filtered/udrl/dst /tmp/udrl_filtered/mones/dst/hypervolume /tmp/udrl_filtered/ra/dst/lr_0.001/population_32/timesteps_100000/ --algo udrl mones ra
    hypervolume DST
    udrl (1e5 steps) : 22845.4 +- 19.200000000000003
    mones (2e5 steps) : 17384.82917280493 +- 6521.100848203363
    ra (1e5 steps) : 22437.4 +- 49.2

    python experiment_plots.py --logs /tmp/udrl_filtered/udrl/minecart /tmp/udrl_filtered/mones/minecart/hypervolume /tmp/udrl_filtered/ra/minecart/lr_0.0003/population_36/timesteps_20000000/ --algo udrl mones ra
    hypervolume Minecart (1e7 steps)
    udrl : 197.56116943359376 +- 0.6956010536777792
    mones: 123.8095121942661 +- 23.031108144462085
    ra : 123.92283817728685 +- 0.2554055852587759

    python experiment_plots.py --logs /tmp/udrl_filtered/udrl/sumo/ /tmp/udrl_filtered/mones/sumo/hypervolume /tmp/udrl_filtered/ra/sumo/lr_0.0003/population_32/timesteps_2000000/03-24/ --algo udrl mones ra 
    hypervolume SUMO (1.5e6 steps)
    udrl : 539.5291748046875 +- 6.274494821585349
    mones : 429.08782958984375 +- 27.474808894745674
    ra : 466.0217782169205 +- 31.22539289605104

    # epsilon from same plots
    =====dst=====
    mean:
    udrl-max      0.039024
    udrl-mean     0.003902
    mones-max     0.686992
    mones-mean    0.364553
    ra-max        0.666667
    ra-mean       0.318591
    std:
    udrl-max      0.087261
    udrl-mean     0.008726
    mones-max     0.221703
    mones-mean    0.240203
    ra-max        0.000000
    ra-mean       0.002485

    =====minecart=====
    mean:
    udrl-max      0.270769
    udrl-mean     0.065198
    mones-max     1.596154
    mones-mean    1.109120
    ra-max        1.000000
    ra-mean       0.630828
    std:
    udrl-max      0.087210
    udrl-mean     0.027908
    mones-max     0.889341
    mones-mean    0.724919
    ra-max        0.000000
    ra-mean       0.003533

    =====sumo=====
    mean:
    udrl-max      0.247462
    udrl-mean     0.033006
    mones-max     0.659882
    mones-mean    0.401732
    ra-max        0.408317
    ra-mean       0.241427
    std:
    udrl-max      0.172027
    udrl-mean     0.020046
    mones-max     0.199576
    mones-mean    0.109045
    ra-max        0.038843
    ra-mean       0.067428
    """
