 # Using Pareto Conditioned Networks for COVID exit strategies

This repo uses the gym environment (https://github.com/plibin/gym-covid) that encapsulates the stochastic compartment model by Abrams et al. (2021), to learn exit strategies of the first COVID wave that occurred in Belgium.

If you use this model, please cite these works:
- [Abrams, S., Wambua, J., Santermans, E., Willem, L., Kuylen, E., Coletti, P., Libin, P., Faes, C., Petrof, O., Herzog, S., Beutels P., Hens, N. (2021). Modelling the early phase of the Belgian COVID-19 epidemic using a stochastic compartmental model and studying its implied future trajectories. Epidemics, 35, 100449.](https://www.sciencedirect.com/science/article/pii/S1755436521000116?via%3Dihub)
- [Reymond, M., Hayes, C. F., Willem, L., Rădulescu, R., Abrams, S., Roijers, D., Howley, E., Mannion, P., Hens, N., Nowé, A., Libin, P. (2024). Exploring the pareto front of multi-objective covid-19 mitigation policies using reinforcement learning. Expert Systems with Applications, 249, 123686.](https://www.sciencedirect.com/science/article/pii/S0957417424005529)

## How to train

Have a look at the different parameters:
```
python main_pcn.py --help
```

For example, to run the binomial env with continuous actions:
```
python main_pcn.py --env binomial --action continuous
```

This will create logs and checkpoints in the `\tmp\pcn` directory.

## How to evaluate

You can visualize the Coverage set and Hypervolume using:
```
python experiment_plots.py <path-to-logdir>
```

You can also evaluate a checkpoint using:
```
python eval_pcn.py <env-type> <path-to-logdir> --interactive
```
