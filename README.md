 # Using Pareto Conditioned Networks for COVID exit strategies

This repo uses the gym_covid compartment model to learn exit strategies of the first COVID wave that occured in Belgium.

Changes compared to the original PCN:
 - added MultiDiscrete and Continuous actions
 - added optional noise to the target-returns for more robust policies
   > TODO smarter way to deal with stochastic envs
 - tiny change in the __dominating score__ metric
   > TODO make this robust to many envs


## How to train

Have a look at the different parameters:
```
python main_pcn.py --help
```

For example, to run the the binomial env with continuous actions:
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

### saved model

There is a saved model in the `runs/` directory. It also contains the coverage set and hypervolume:

![the coverage set](runs/commit_4c805a2e2cffcfeaf853f91a80c872b2f4288e65/env_binomial/action_continuous/lr_0.001/steps_300000.0/batch_256/model_updates_50/top_episodes_200/n_episodes_10/er_size_500/threshold_0.02/noise_0.1/model_conv1d/2022-02-28_12-12-38_66aa/plots/pf.png)

![the hypervolume](runs/commit_4c805a2e2cffcfeaf853f91a80c872b2f4288e65/env_binomial/action_continuous/lr_0.001/steps_300000.0/batch_256/model_updates_50/top_episodes_200/n_episodes_10/er_size_500/threshold_0.02/noise_0.1/model_conv1d/2022-02-28_12-12-38_66aa/plots/hv.png)

Have a look at the executed policies in
```
runs/commit_4c805a2e2cffcfeaf853f91a80c872b2f4288e65/env_binomial/action_continuous/lr_0.001/steps_300000.0/batch_256/model_updates_50/top_episodes_200/n_episodes_10/er_size_500/threshold_0.02/noise_0.1/model_conv1d/2022-02-28_12-12-38_66aa/policy-executions/
```

<figure class="video_container">
  <video controls="true" allowfullscreen="true" poster="runs/commit_4c805a2e2cffcfeaf853f91a80c872b2f4288e65/env_binomial/action_continuous/lr_0.001/steps_300000.0/batch_256/model_updates_50/top_episodes_200/n_episodes_10/er_size_500/threshold_0.02/noise_0.1/model_conv1d/2022-02-28_12-12-38_66aa/policy-executions/policy_0.png">
    <source src="runs/commit_4c805a2e2cffcfeaf853f91a80c872b2f4288e65/env_binomial/action_continuous/lr_0.001/steps_300000.0/batch_256/model_updates_50/top_episodes_200/n_episodes_10/er_size_500/threshold_0.02/noise_0.1/model_conv1d/2022-02-28_12-12-38_66aa/policy-executions/all_policies.mp4" type="video/mp4">
  </video>
</figure>

Or try out the policies directly using:
```
python eval_pcn.py binomial runs/commit_4c805a2e2cffcfeaf853f91a80c872b2f4288e65/env_binomial/action_continuous/lr_0.001/steps_300000.0/batch_256/model_updates_50/top_episodes_200/n_episodes_10/er_size_500/threshold_0.02/noise_0.1/model_conv1d/2022-02-28_12-12-38_66aa/ --n 10 --interactive
```