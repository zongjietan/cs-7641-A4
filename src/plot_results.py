"""Plot experiment results.

"""
import numpy as np
import matplotlib
matplotlib.use('agg')
import seaborn as sns
from matplotlib import pyplot as plt
from helpers import get_abspath
sns.set_style('darkgrid')


def get_annotations(x):
    """Converts integers in Frozen Lake policy to arrows indicating direction.

    Args:
        x (numpy.array): Policy array.

    """
    y = np.chararray(x.shape, unicode=True)
    y[:] = 'x'
    y[x == 0] = '<'
    y[x == 1] = 'V'
    y[x == 2] = '>'
    y[x == 3] = '^'

    return y


def plot_grid(heat, prefix, tdir, big=False, policy_for_annot=None):
    """Plots grid using a scalar-based heatmap.

    Args:
        heat (numpy.array): Heat map values.
        prefix (str): Prefix for CSV and plot outputs.
        tdir (str): Target directory.
        policy_for_annot (numpy.array): Policy array to use for annotations.

    """
    figsize = (5, 5)
    if heat.shape[0] > 10:
        figsize = (8, 8)
    if policy_for_annot is not None:
        policy_for_annot = get_annotations(policy_for_annot)

    # generate graph
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    ax = sns.heatmap(heat, annot=policy_for_annot, fmt='')
    ax.set_title('Optimal Policy ({})'.format(prefix), fontsize=20)
    fig.tight_layout()

    # save figure
    plotpath = get_abspath('{}_policygrid.png'.format(prefix), tdir)
    plt.savefig(plotpath)
    plt.close()


def plot_reward_curve(rewards, prefix, tdir, xlabel='Iterations'):
    """Plots rewards as a function of number of iterations or episodes.

    Args:
        rewards (pandas.dataframe): Rewards dataframe.
        prefix (str): Prefix for CSV and plot outputs.
        tdir (str): Target directory.

    """
    # get figure and axes
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))

    # plot reward curve
    k = rewards['k']
    r = rewards['r']
    ax.plot(k, r, color='b')
    ax.set_title('Average Rewards ({})'.format(prefix))
    ax.set_ylabel('Average Reward')
    ax.set_xlabel('Episodes')
    ax.grid(linestyle='dotted')
    fig.tight_layout()

    # save figure
    plotpath = get_abspath('{}_rewards.png'.format(prefix), tdir)
    plt.savefig(plotpath)
    plt.close()


def plot_delta_curve(deltas, prefix, tdir):
    """Plots delta as a function of number of episodes.

    Args:
        rewards (pandas.dataframe): Rewards dataframe.
        prefix (str): Prefix for CSV and plot outputs.
        tdir (str): Target directory.

    """
    # get figure and axes
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))

    # plot reward curve
    k = deltas['k']
    d = deltas['d']
    ax.plot(k, d, color='g')
    ax.set_title('Delta Convergence ({})'.format(prefix))
    ax.set_ylabel('Delta Value')
    ax.set_xlabel('Episodes')
    ax.grid(linestyle='dotted')
    fig.tight_layout()

    # save figure
    plotpath = get_abspath('{}_deltas.png'.format(prefix), tdir)
    plt.savefig(plotpath)
    plt.close()


def plot_reward_combined(df1, df2, df3, prefix, tdir):
    """Plot combined rewards chart for all values of gamma.

    Args:
        df1 (pandas.dataframe): Gamma 0.1.
        df2 (pandas.dataframe): Gamma 0.9.
        df3 (pandas.dataframe): Gamma 0.99.
        prefix (str): Prefix for CSV and plot outputs.
        tdir (str): Target directory.

    """
    # get figure and axes
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))

    # plot reward curve
    k = df1['k']
    r1 = df1['r']
    r2 = df2['r']
    r3 = df3['r']
    ax.plot(k, r1, color='b', label='Gamma 0.1')
    ax.plot(k, r2, color='r', label='Gamma 0.9')
    ax.plot(k, r3, color='g', label='Gamma 0.99')
    ax.set_title('Rewards per Episode ({})'.format(prefix))
    ax.set_ylabel('Average Reward')
    ax.set_xlabel('Episodes')
    ax.legend(loc='best')
    ax.grid(linestyle='dotted')
    fig.tight_layout()

    # save figure
    plotpath = get_abspath('{}_combined_rewards.png'.format(prefix), tdir)
    plt.savefig(plotpath)
    plt.close()


def plot_delta_combined(df1, df2, df3, prefix, tdir):
    """Plot combined delta chart for all values of gamma.

    Args:
        df1 (pandas.dataframe): Gamma 0.1.
        df2 (pandas.dataframe): Gamma 0.9.
        df3 (pandas.dataframe): Gamma 0.99.
        prefix (str): Prefix for CSV and plot outputs.
        tdir (str): Target directory.

    """
    # get figure and axes
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))

    # plot reward curve
    k = df1['k']
    r1 = df1['d']
    r2 = df2['d']
    r3 = df3['d']
    ax.plot(k, r1, color='b', label='Gamma 0.1')
    ax.plot(k, r2, color='r', label='Gamma 0.9')
    ax.plot(k, r3, color='g', label='Gamma 0.99')
    ax.set_title('Delta Convergence ({})'.format(prefix))
    ax.set_ylabel('Delta')
    ax.set_xlabel('Episodes')
    ax.legend(loc='best')
    ax.grid(linestyle='dotted')
    fig.tight_layout()

    # save figure
    plotpath = get_abspath('{}_combined_deltas.png'.format(prefix), tdir)
    plt.savefig(plotpath)
    plt.close()
