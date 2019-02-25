from scripts.markovDecisionProcess import MarkovDecisionProcess
from scripts.qlearning import Qlearning
from scripts.learningAlgorithm import LearningAlgorithm


class main:

    mdp = MarkovDecisionProcess(0, range(16), range(4))
    strategy = Qlearning(mdp, 1, 0.1, 1.0,1.0, 0.01)
    algorithm = LearningAlgorithm(strategy, 'FrozenLake-v0')
    algorithm.learn(1000)
