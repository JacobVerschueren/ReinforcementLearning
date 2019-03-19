from scripts.learning_strategies.learningStrategy import LearningStrategy
from scripts.percept import Percept
import numpy as np
import math
from numba import jit


class ValueIteration(LearningStrategy):
    """
    Value Iteration subclass of the learning strategy
    """

    def __init__(self, mdp, learning_rate, decay_rate, precision, gamma, epsilon_max = 1.0, epsilon_min = 0.01):
        LearningStrategy.__init__(self, mdp, learning_rate, decay_rate, gamma, epsilon_max, epsilon_min)
        self.states = mdp.get_states()
        self.n_states = len(self.states)
        self.actions = mdp.get_actions()
        self.n_actions = len(self.actions)
        self.ζ = precision

    # @jit
    def evaluate(self, p: Percept):
        self.mdp.update(p)
        r_max = np.max(self.mdp.rsa)
        Δ = math.inf

        while Δ > (self.ζ * r_max * (1 - self.γ)/self.γ):
            Δ = 0
            for s in range(self.n_states):
                u = self.v_values[s]
                self.v_values[s] = np.max(self.value_function(s))
                Δ = max(Δ, abs(u - self.v_values[s]))

    # @jit
    def value_function(self, s):
        eu = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            asum = 0
            for s2 in range(self.n_states):
                asum += (self.mdp.ptsa[s, a, s2] * (self.mdp.get_specific_rsa(s, a) + (self.γ * self.v_values[s2])))
            asum *= self.policy[s, a]
            eu[a] = asum
        return eu
