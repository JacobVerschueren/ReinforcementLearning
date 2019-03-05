import numpy as np
import abc
import math
from scripts.markovDecisionProcess import MarkovDecisionProcess

class LearningStrategy(abc.ABC):

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
        self.γ = 0.4 #TODO is this ok?
        self.epsilon_seq = [1.0]  # for epsilon decay visualisation

    def learn(self, percept, episode_num):
        self.evaluate(percept)
        self.improve(episode_num)

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
            # print(optimal_action)

            for a in range(self.n_actions):
                if optimal_action == a:
                    self.policy[s, a] = 1 - self.ε + (self.ε / self.n_actions)
                else:
                    self.policy[s, a] = self.ε / self.n_actions

        old_random = self.ε
        self.ε = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * math.exp((-self.λ) * episode_num)

        if old_random != self.ε:
         self.epsilon_seq.append(self.ε)
