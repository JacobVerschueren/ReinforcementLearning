import numpy as np
import abc
import math
from scripts.markovDecisionProcess import MarkovDecisionProcess


class LearningStrategy(abc.ABC):
    """
    superclass of the different learning algorithms
    improve method only works when v-values are used, if anything else is needed this needs to be adjusted
    """

    def __init__(self, mdp: MarkovDecisionProcess, learning_rate, decay_rate, epsilon_max = 1.0, epsilon_min = 0.01):
        self.α = learning_rate
        self.λ = decay_rate
        self.ε = epsilon_max
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.mdp = mdp
        self.n_states = len(self.mdp.states)
        self.n_actions = len(self.mdp.actions)
        self.policy = np.full((len(mdp.states), len(mdp.actions)), 1/self.n_actions)
        self.v_values = np.zeros((len(mdp.states)))
        self.γ = 0.6
        # self.epsilon_seq = [1.0]  # uncomment for epsilon decay visualisation
        self.episode_count = 0

    def learn(self, percept):
        self.evaluate(percept)
        self.improve(self.episode_count)
        self.episode_count += 1

    @abc.abstractmethod
    def evaluate(self, percept):
        pass

    # @abc.abstractmethod
    def improve(self, episode_num):
        for s in range(self.n_states):
            action_values = np.zeros(self.n_actions)
            for a in range(self.n_actions):
                action_value = 0
                rsa = self.mdp.get_specific_rsa(s, a)
                for s2 in range(self.n_states):
                    action_value += (self.mdp.get_specific_ptsa(s, a, s2) * (rsa + self.γ * self.v_values[s2]))
                action_values[a] = action_value

            optimal_action = np.random.choice(np.where(action_values == action_values.max())[0])

            for a in range(self.n_actions):
                if optimal_action == a:
                    self.policy[s, a] = 1 - self.ε + (self.ε / self.n_actions)
                else:
                    self.policy[s, a] = self.ε / self.n_actions

        # old_random = self.ε #uncomment for epsilon decay visualisation
        if self.ε > self.epsilon_min:
            self.ε = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * math.exp((-self.λ) * episode_num)

        """
        # uncomment for epsilon decay visualisation
        if old_random != self.ε:
         self.epsilon_seq.append(self.ε)
        """