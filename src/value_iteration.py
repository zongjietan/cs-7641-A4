"""Value iteration experiments with Frozen Lake.

"""
import gym
import numpy as np
import pandas as pd
from helpers import get_abspath, timing
from helpers import evaluate_rewards_and_transitions
from plot_results import plot_grid
from gym.envs.registration import register

# register custom Frozen Lake 5x5 map
SIZE = 5
MAP_FILE = get_abspath('frozen_lake_{}.txt'.format(SIZE), 'data')
with open(MAP_FILE) as f:
    MAP_DATA = f.readlines()
MAP5 = [x.strip() for x in MAP_DATA]

# register custom Frozen Lake 20x20 map
SIZE = 20
MAP_FILE = get_abspath('frozen_lake_{}.txt'.format(SIZE), 'data')
with open(MAP_FILE) as f:
    MAP_DATA = f.readlines()
MAP20 = [x.strip() for x in MAP_DATA]

register(
    id='FrozenLake5x5-v1',
    entry_point='frozen_lake_custom:FrozenLakeCustom',
    max_episode_steps=400,
    reward_threshold=0.99,  # optimum = 1
    kwargs={'desc': MAP5, 'goalr': 1.0, 'fallr': 0.0, 'stepr': 0.0}
)

register(
    id='FrozenLake20x20-v1',
    entry_point='frozen_lake_custom:FrozenLakeCustom',
    max_episode_steps=5000,
    reward_threshold=0.99,  # optimum = 1
    kwargs={'desc': MAP20, 'goalr': 1.0, 'fallr': 0.0, 'stepr': 0.0}
)

register(
    id='FrozenLake5x5Neg-v1',
    entry_point='frozen_lake_custom:FrozenLakeCustom',
    max_episode_steps=400,
    reward_threshold=0.99,  # optimum = 1
    kwargs={'desc': MAP5, 'goalr': 1000.0,
            'fallr': -1000.0, 'stepr': -1.0}
)

register(
    id='FrozenLake20x20Neg-v1',
    entry_point='frozen_lake_custom:FrozenLakeCustom',
    max_episode_steps=5000,
    reward_threshold=0.99,  # optimum = 1
    kwargs={'desc': MAP20, 'goalr': 1000.0, 'fallr': -1000.0, 'stepr': -1.0}
)


@timing
def value_iteration(problem, R=None, T=None, gamma=0.99, max_iterations=10**7, delta=10**-4):
    """ Runs Value Iteration on a gym problem.

    Args:
        problem (gym.env): Gym problem object.
        R (numpy.array): Rewards matrix.
        T (numpy.array): Transition matrix.
        gamma (float): Discount rate.
        max_iterations (float): Maximum number of iterations.
        delta (float): Convergence threshold.
    Returns:
        policy (numpy.array): Optimal policy.
        rewards (list(float)): Start state value function for each iteration.
        iters (int): Number of iterations until convergence.
        deltas (list(float)): Diff between current and prior value function.
        value_fn (numpy.array): Value function array.

    """
    value_fn = np.zeros(problem.observation_space.n)
    rewards = []
    deltas = []
    iters = 0
    if R is None or T is None:
        R, T = evaluate_rewards_and_transitions(problem)

    for i in range(max_iterations):
        previous_value_fn = value_fn.copy()
        Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value_fn)
        value_fn = np.max(Q, axis=1)
        rewards.append(value_fn[0])
        iters += 1
        curr_delta = np.max(np.abs(value_fn - previous_value_fn))
        deltas.append(curr_delta)
        if curr_delta < delta:
            break

    # return optimal policy, rewards and iterations
    policy = np.argmax(Q, axis=1)
    return policy, rewards, iters, deltas, value_fn


def run_experiment(problem, prefix, gamma, shape=None):
    """Run value iteration experiment.

    Args:
        problem (str): Gym problem name.
        prefix (str): Prefix for CSV and plot outputs.
        gamma (float): Gamma value.
        shape (tuple(int)): Shape of state space array.

    """
    problem = gym.make(problem)
    policy, rewards, iters, deltas, value_fn = value_iteration(
        problem, gamma=gamma)
    idxs = [i for i in range(0, iters)]
    print('{}: {} iterations to converge'.format(prefix, iters))

    # save results as CSV
    resdir = 'results/VI'
    q = get_abspath('{}_policy.csv'.format(prefix), resdir)
    r = get_abspath('{}_rewards.csv'.format(prefix), resdir)
    d = get_abspath('{}_deltas.csv'.format(prefix), resdir)
    v = get_abspath('{}_value_fn.csv'.format(prefix), resdir)
    pdf = pd.DataFrame(policy)
    rdf = pd.DataFrame(np.column_stack([idxs, rewards]), columns=['k', 'r'])
    ddf = pd.DataFrame(np.column_stack([idxs, deltas]), columns=['k', 'd'])
    vdf = pd.DataFrame(value_fn)
    pdf.to_csv(q, index=False)
    rdf.to_csv(r, index=False)
    ddf.to_csv(d, index=False)
    vdf.to_csv(v, index=False)

    # plot results
    tdir = 'plots/VI'
    polgrid = pdf.as_matrix().reshape(shape)
    heatmap = vdf.as_matrix().reshape(shape)
    plot_grid(heatmap, prefix, tdir, policy_for_annot=polgrid)

    return iters


def main():
    """Run simulation.

    """
    
    gammas = [0.001, 0.25, 0.5, 0.75, 0.9, 0.99, 0.995, 0.999]
    shapes = [(5, 5), (20, 20)]
    shape_map = {(5, 5): 'FrozenLake5x5-v1', (20, 20): 'FrozenLake20x20-v1'}

    for shape in shapes:
        problem = shape_map[shape]

        iterToConverge = []
        for gamma in gammas:
            prefix = 'fl_vi_{}x{}_{}'.format(shape[0], shape[1], gamma)
            iterToConverge.append(run_experiment(problem, prefix, gamma, shape))

        print(iterToConverge)
    
    shape_map = {(5, 5): 'FrozenLake5x5Neg-v1', (20, 20): 'FrozenLake20x20Neg-v1'}

    for shape in shapes:
        problem = shape_map[shape]

        iterToConverge = []

        for gamma in gammas:
            prefix = 'fl_vi_{}x{}Neg_{}'.format(shape[0], shape[1], gamma)
            iterToConverge.append(run_experiment(problem, prefix, gamma, shape))

        print(iterToConverge)

if __name__ == '__main__':
    main()
