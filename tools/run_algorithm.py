"""
Main script that runs all of the algorithms on data files for each of the
objects in Figure 1 of the paper. Saves learning curve plots.
"""

import os
import numpy as np
import tqdm
import argparse
import yaml

from grasp_exploration import GQCNNPolicy, UCB, TS, OptimalPolicy, StablePoseEnv
from UCRL import UcrlMdp

import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_context('poster')
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


def get_alg(name, params, num_arms, env):
    if name == 'GQCNN':
        return GQCNNPolicy(num_arms, **params)
    elif name == 'UCRL2':
        return UcrlMdp(env, r_max=1, random_state=np.random.randint(2 ** 30))
    elif name == 'UCB':
        return UCB(num_arms, **params)
    elif name == 'TS':
        return TS(num_arms, **params)
    elif name == 'Optimal':
        return OptimalPolicy(num_arms, env)
    else:
        raise ValueError('Unsupported algorithm {}'.format(name))


if __name__ == "__main__":

    # Script arguments
    parser = argparse.ArgumentParser(description='Evaluate Grasp Exploration Algorithms')
    parser.add_argument('obj_data_path', type=str, help='path to object data file')
    parser.add_argument('--cfg', type=str, default='cfg/run_algorithm.yaml',
                        help='config file with evaluation params')

    args = parser.parse_args()
    obj_data = np.load(args.obj_data_path)
    obj_name = os.path.basename(args.obj_data_path)[:-4]
    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    num_rollouts = cfg["num_rollouts"]
    rollout_steps = cfg['rollout_steps']

    policies = [{'type': 'Optimal', 'params' : {}}] + cfg['policies']
    num_pols = len(policies)

    # Extract data for env
    stp_probs = obj_data["probs"]
    arm_means = obj_data["gt_values"]
    priors = obj_data["q_values"]
    num_trials, num_poses, num_arms = priors.shape

    topple_matrix = np.eye(len(stp_probs))
    if cfg["toppling"]:
        topple_matrix = obj_data["topple_matrix"]

    # Track rewards
    rewards = np.zeros((len(policies), num_trials,
                        num_rollouts, rollout_steps), dtype=np.bool)

    # Iterate over trials
    for i in tqdm.trange(num_trials, desc='Trial'):
        env = StablePoseEnv(arm_means[i],
                            stp_probs,
                            topple_matrix)

        # Iterate over rollouts
        for j, pol in enumerate(policies):
            pbar = tqdm.trange(num_rollouts, leave=False)
            for k in pbar:
                pbar.set_description('{}'.format(pol['type']))
                alg = get_alg(pol['type'], pol['params'], (num_poses, num_arms), env)
                alg.set_prior(priors[i])
                env.reset()

                if isinstance(alg, UcrlMdp):
                    alg_rewards, alg_poses = alg.learn(rollout_steps, 5000)
                    rewards[j, i, k] = alg_rewards
                else:
                    ts = 0
                    while ts < rollout_steps:
                        pose = env.pose
                        arm = alg.select_arm(pose)
                        reward = env.step(arm)
                        alg.update(pose, arm, reward)
                        rewards[j, i, k, ts] = reward
                        ts += 1

    # Calculate average rewards
    N_avg = 20
    cum_rewards = rewards.cumsum(axis=-1)
    avg_rewards = (cum_rewards[..., N_avg:] - cum_rewards[..., :-N_avg]) / float(N_avg)

    # Create dataframe for plotting
    df = pd.DataFrame()
    df['Timestep $(t)$'] = np.tile(np.arange(rollout_steps - N_avg), num_trials * num_rollouts * num_pols)
    df['Reward'] = avg_rewards.flatten()
    policy_names = []
    for p in cfg["policies"]:
        if p["type"] == 'UCB' or p["type"] == "TS":
            if p["params"]["strength"] > 0:
                policy_names.append('BORGES ({}-{})'.format(p["type"], p["params"]["strength"]))
            else:
                policy_names.append('BORGES ({})'.format(p["type"]))
        else:
            policy_names.append(p["type"])
    policy_names = ["Optimal"] + policy_names
    df['Policy'] = np.repeat(policy_names, num_rollouts * num_trials * (rollout_steps - N_avg))

    # Generate plot of reward curves and save
    plt.figure()
    ci_fmt = 95 if cfg["plot_ci"] else None
    ax = sns.lineplot(x='Timestep $(t)$', y='Reward',
                        hue='Policy', data=df, ci=ci_fmt)

    plt.ylim([-0.01, 1.01])
    plt.xlim([-1, rollout_steps + 1])
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.savefig('{}_reward.png'.format(obj_name),
                bbox_inches="tight", dpi=300)
    plt.close()

