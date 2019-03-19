from scripts.markovDecisionProcess import MarkovDecisionProcess
from scripts.learning_strategies.qlearning import Qlearning
from scripts.learning_strategies.nstepqlearning import NstepQlearning
from scripts.learning_strategies.monteCarlo import MonteCarlo
from scripts.agent import Agent


class Main:
    """
    Solely for testing purposes. Do not use.
    """

    mdp = MarkovDecisionProcess(0, range(16), range(4))
    strategy = Qlearning(mdp, 1, 0.1, 1.0, 0.01)
    algorithm = Agent(strategy, 'FrozenLake-v0')
    algorithm.learn(1000)

    """
    Use code below to visualise evolution of epsilon
    When not needed comment out as well as code in learningstrategy.py
    """

    """
    s = strategy.epsilon_seq
    t = np.arange(0.0, len(s))

    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='episodes', ylabel='epsilon',
           title='epsilon over time')
    ax.grid()

    plt.show()
    """

