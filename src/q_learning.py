"""Q-Learning experiments with Frozen Lake.

"""
import math
import gym
import numpy as np
import pandas as pd
import os
import time
from helpers import get_abspath, timing
from plot_results import plot_grid
from gym.envs.registration import register


# register custom Frozen Lake 25x25 map
SIZE = 50
MAP_FILE = get_abspath('frozen_lake_{}.txt'.format(SIZE), 'data')
with open(MAP_FILE) as f:
    MAP_DATA = f.readlines()
MAP50 = [x.strip() for x in MAP_DATA]

register(
    id='FrozenLake8x8-v1',
    entry_point='frozen_lake_custom:FrozenLakeCustom',
    max_episode_steps=5000,
    reward_threshold=0.99,  # optimum = 1
    kwargs={'map_name': '8x8', 'goalr': 1.0, 'fallr': 0.0, 'stepr': 0.0}
)

register(
    id='FrozenLake50x50-v1',
    entry_point='frozen_lake_custom:FrozenLakeCustom',
    max_episode_steps=5000,
    reward_threshold=0.99,  # optimum = 1
    kwargs={'desc': MAP50, 'goalr': 1.0, 'fallr': 0.0, 'stepr': 0.0}
)

register(
    id='FrozenLake8x8Neg-v1',
    entry_point='frozen_lake_custom:FrozenLakeCustom',
    max_episode_steps=5000,
    reward_threshold=0.99,  # optimum = 1
    kwargs={'map_name': '8x8', 'goalr': 1000.0,
            'fallr': -1000.0, 'stepr': -1.0}
)

register(
    id='FrozenLake50x50Neg-v1',
    entry_point='frozen_lake_custom:FrozenLakeCustom',
    max_episode_steps=5000,
    reward_threshold=0.99,  # optimum = 1
    kwargs={'desc': MAP50, 'goalr': 1000.0, 'fallr': -1000.0, 'stepr': -1.0}
)


def d1(e):
    """Returns harmonic schedule for random action rate decay.

    Args:
        e (int): Episode number.
    Returns:
        d (float): Random action rate.

    """
    d = 1 / float(e + 1)
    return d


def d2(e, rar=0.99, radr=0.9):
    """Returns logarithmic decay schedule for random action rate decay.

    Args:
        e (int): Episode number.
        rar (float): Random action rate initial value.
        radr (float): Decay rate.
    Returns:
        rar (float): Random action rate.

    """
    for i in range(0, e + 1):
        rar *= radr
    return rar


def d3(e):
    """Returns exponential decay schedule for random action rate decay.

    Args:
        e (int): Episode number.
    Returns:
        d (float): Random action rate.

    """
    d = float(math.e**(-e * 0.0005))
    return d


@timing
def q_learning(env, alpha=0.75, decay=d1, gamma=0.99, episodes=5000):
    """ Runs Q-Learning on a gym problem.

    Args:
        env (gym.env): Gym problem object.
        decay (function): Decay function for random action rate.
        alpha (float): Learning rate.
        gamma (float): Discount rate.
        episodes (int): Number of episodes.
    Returns:
        policy (numpy.array): Optimal policy.
        i + 1 (int): Number of iterations until convergence.

    """
    # Q, maximum steps, visits and rewards
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []
    iterations = []
    max_steps = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
    visits = np.zeros((env.observation_space.n, 1))

    # episodes
    for episode in range(episodes):
        # refresh state
        state = env.reset()
        done = False
        t_reward = 0

        # run episode
        for i in range(max_steps):
            if done:
                break

            current = state
            action = np.argmax(Q[current, :] +
                               np.random.randn(1, env.action_space.n) * decay(episode))

            state, reward, done, info = env.step(action)
            visits[state] += 1
            t_reward += reward
            Q[current, action] += alpha * \
                (reward + gamma * np.max(Q[state, :]) - Q[current, action])

        rewards.append(t_reward)
        iterations.append(i)

    return Q, rewards, visits


def chunk_list(l, n):
    """Used to create rolling average by split up a list into chunks and
    averaging the values in each chunk.

    Args:
        l (list): List object.
        n (int): Step size.
    Yields:
        list of length n with average values across each chunk.

    """
    for i in range(0, int(len(l)), n):
        yield l[i:i + n]


def run_experiment(problem, prefix, alpha, gamma, d, shape=None):
    """Run Q-Learning experiment for specified Gym problem and write results
    to CSV files.

    Args:
        problem (str): Gym problem name.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        d (function): Epsilon decay function.
        shape (tuple(int)): Shape of state space matrix.
        prefix (str): Prefix for CSV and plot outputs.

    """
    episodes = 5000
    size = episodes // 100

    # instantiate environment and run Q-learner
    start = time.time()
    env = gym.make(problem)
    Q, rewards, visits = q_learning(env, alpha, d, gamma)
    env.close()
    end = time.time()
    elapsed = end - start

    # average rewards
    k = [i for i in range(0, episodes, size)]
    chunks = list(chunk_list(rewards, size))
    rewards = [sum(chunk) / len(chunk) for chunk in chunks]

    # save results as CSV
    resdir = 'results/QL'
    qf = get_abspath('{}_policy.csv'.format(prefix), resdir)
    rf = get_abspath('{}_rewards.csv'.format(prefix), resdir)
    vf = get_abspath('{}_visits.csv'.format(prefix), resdir)
    qdf = pd.DataFrame(Q)
    vdf = pd.DataFrame(visits)
    rdf = pd.DataFrame(np.column_stack([k, rewards]), columns=['k', 'r'])
    qdf.to_csv(qf, index=False)
    vdf.to_csv(vf, index=False)
    rdf.to_csv(rf, index=False)

    # write timing results and average reward in last iteration
    combined = get_abspath('summary.csv', 'results/QL')
    with open(combined, 'a') as f:
        f.write('{},{},{}\n'.format(prefix, elapsed, rdf.iloc[-1, 1]))

    # plot results
    tdir = 'plots/QL'
    polgrid = qdf.as_matrix().argmax(axis=1).reshape(shape)
    heatmap = vdf.as_matrix().reshape(shape)
    plot_grid(heatmap, prefix, tdir, policy_for_annot=polgrid)


def main():
    """Run experiments.

    """
    e_decay = [d1, d2, d3]
    alphas = [0.01, 0.25, 0.5, 0.75, 0.99]
    gammas = [0.001, 0.9, 0.99]
    shapes = [(8, 8), (50, 50)]

    # normal envs
    shape_map = {(8, 8): 'FrozenLake8x8-v1', (50, 50): 'FrozenLake50x50-v1'}

    # reinitialize summary file
    try:
        combined = get_abspath('summary.csv', 'results/QL')
        os.remove(combined)
    except Exception:
        pass

    with open(combined, 'a') as f:
        f.write('{},{},{}\n'.format('func', 'time', 'avg_reward'))

    for shape in shapes:
        problem = shape_map[shape]
        for gamma in gammas:
            for alpha in alphas:
                for d in e_decay:
                    prefix = 'fl_ql_{}x{}_{}_{}_{}'.format(
                        shape[0], shape[1], gamma, alpha, d.__name__)
                    run_experiment(problem, prefix, alpha, gamma, d, shape)

    # negative envs
    e_decay = [d1, d3]
    shape_map = {(8, 8): 'FrozenLake8x8Neg-v1',
                 (50, 50): 'FrozenLake50x50Neg-v1'}

    for shape in shapes:
        problem = shape_map[shape]
        for gamma in gammas:
            for alpha in alphas:
                for d in e_decay:
                    prefix = 'fl_ql_{}x{}neg_{}_{}_{}'.format(
                        shape[0], shape[1], gamma, alpha, d.__name__)
                    run_experiment(problem, prefix, alpha, gamma, d, shape)


if __name__ == '__main__':
    main()
