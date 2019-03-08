from scripts.learningStrategy import LearningStrategy
from scripts.percept import Percept
import numpy as np


class Qlearning(LearningStrategy):
    """
    qlearning subclass of the learning strategy
    """

    def __init__(self, mdp, learning_rate, decay_rate, epsilon_max = 1.0, epsilon_min = 0.01):
        LearningStrategy.__init__(self, mdp, learning_rate, decay_rate, epsilon_max, epsilon_min)
        self.states = mdp.get_states()
        self.n_states = len(self.states)
        self.actions = mdp.get_actions()
        self.n_actions = len(self.actions)
        self.qvalues = np.zeros((self.n_states, self.n_actions))

    def evaluate(self, p: Percept):
        self.mdp.update(p)
        max_qa = np.max(self.qvalues[p.new_state])
        self.qvalues[p.old_state, p.action] += \
            (self.α * (self.mdp.get_specific_rsa(p.old_state, p.action) + self.γ *
                       (max_qa - self.qvalues[p.old_state, p.action])))

        for s in range(self.n_states):
            self.v_values[s] = np.max(self.qvalues[s])
