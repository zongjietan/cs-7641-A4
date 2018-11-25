"""Helper functions.

"""
from functools import wraps
import os
import time
import numpy as np


def save_array(array, filename, sep=',', subdir='data'):
    """Saves a Numpy array as a delimited text file.

    Args:
        array (Numpy.Array): Input array.
        filename (str): Output file name.
        sep (str): Delimiter.
        subdir (str): Parent directory path for output file.

    """
    tdir = os.path.join(os.getcwd(), os.pardir, subdir, filename)
    np.savetxt(fname=tdir, X=array, delimiter=sep, fmt='%.20f')


def save_dataset(df, filename, sep=',', subdir='data', header=True):
    """Saves Pandas data frame as a CSV file.

    Args:
        df (Pandas.DataFrame): Data frame.
        filename (str): Output file name.
        sep (str): Delimiter.
        subdir (str): Project directory to save output file.
        header (Boolean): Specify inclusion of header.

    """
    tdir = os.path.join(os.getcwd(), os.pardir, subdir, filename)
    df.to_csv(path_or_buf=tdir, sep=sep, header=header, index=False)


def get_abspath(filename, filepath):
    """Gets absolute path of specified file within the project directory. The
    filepath has to be a subdirectory within the main project directory.

    Args:
        filename (str): Name of specified file.
        filepath (str): Subdirectory of file.
    Returns:
        fullpath (str): Absolute filepath.

    """
    curpath = os.path.abspath(os.path.join(os.curdir, os.pardir))
    fullpath = os.path.join(curpath, filepath, filename)

    return fullpath


def timing(f):
    """Decorator function to time function execution time.

    Args:
        f (function): Function to be timed.
    Returns:
        wrapper: Function with timing results.

    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        print('{} function took {} ms'.format(f.__name__, elapsed * 1000.0))
        return ret
    return wrapper


def evaluate_rewards_and_transitions(problem, mutate=False):
    """Evaluation of rewards and transitions for policy iteration and value
    iteration.

    Args:
        problem (gym.env): Gym problem.
        mutate (boolean): Determines whether to mutate matrices.
    Returns:
        R (numpy.array): Reward matrix.
        T (numpy.array): Transition matrix.

    """
    # enumerate state and action space sizes
    num_states = problem.observation_space.n
    num_actions = problem.action_space.n

    # initialize T and R matrices
    R = np.zeros((num_states, num_actions, num_states))
    T = np.zeros((num_states, num_actions, num_states))

    # iterate over states, actions, and transitions
    for state in range(num_states):
        for action in range(num_actions):
            for transition in problem.env.P[state][action]:
                probability, next_state, reward, done = transition
                R[state, action, next_state] = reward
                T[state, action, next_state] = probability

            # normalize T across state + action axes
            T[state, action, :] /= np.sum(T[state, action, :])

    # conditionally mutate and return
    if mutate:
        problem.env.R = R
        problem.env.T = T
    return R, T


def print_policy(policy, mapping=None, shape=(0,)):
    """Prints policy table by reshaping

    Args:
        policy (Numpy.Array): Policy array, flattened.
        mapping
    """
    print(np.array([mapping[action] for action in policy]).reshape(shape))
