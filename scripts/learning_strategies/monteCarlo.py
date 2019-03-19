from scripts.learning_strategies.learningStrategy import LearningStrategy
from scripts.percept import Percept
import numpy as np


class MonteCarlo(LearningStrategy):
    """
    Monte Carlo subclass of the learning strategy
    """

    def __init__(self, mdp, learning_rate, decay_rate, gamma, epsilon_max = 1.0, epsilon_min = 0.01):
        LearningStrategy.__init__(self, mdp, learning_rate, decay_rate, gamma, epsilon_max, epsilon_min)
        self.states = mdp.get_states()
        self.actions = mdp.get_actions()
        self.qvalues = np.zeros((len(self.states), len(self.actions)))
        self.percept_list = []

    def evaluate(self, percept: Percept):
        self.mdp.update(percept)
        self.percept_list.insert(0, percept)
        n_percepts = len(self.percept_list)
        plist = self.percept_list

        if percept.final_state:
            for i in range(n_percepts):
                max_qa = np.max(self.qvalues[plist[i].new_state])
                self.qvalues[plist[i].old_state, plist[i].action] -= \
                    (self.α * (self.qvalues[plist[i].old_state, plist[i].action] -
                               (self.mdp.get_specific_rsa(plist[i].old_state, plist[i].action) + self.γ * max_qa)))

        for s in range(len(self.states)):
            self.v_values[s] = np.max(self.qvalues[s])

        if percept.final_state:
            self.percept_list = []
