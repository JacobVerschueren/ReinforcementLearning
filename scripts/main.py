from scripts.markovDecisionProcess import MarkovDecisionProcess
from scripts.qlearning import Qlearning
from scripts.learningAlgorithm import LearningAlgorithm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class main:

    mdp = MarkovDecisionProcess(0, range(16), range(4))
    strategy = Qlearning(mdp, 1, 1,1.0, 0.01)
    algorithm = LearningAlgorithm(strategy, 'FrozenLake-v0')
    algorithm.learn(100)

    """
    Use code below to visualise evolution of epsilon
    When not needed comment out as well as code in learningstrategy.py
    """
    s = strategy.epsilon_seq
    t = np.arange(0.0, len(s))

    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='episodes', ylabel='epsilon',
           title='epsilon over time')
    ax.grid()

    plt.show()

