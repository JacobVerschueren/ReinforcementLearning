import numpy as np
import abc
import math
from scripts.markovDecisionProcess import MarkovDecisionProcess

class LearningStrategy(abc.ABC):

    def __init__(self, mdp: MarkovDecisionProcess, learning_rate, decay_rate, epsilon_max = 1.0, epsilon_min = 0.01):
        self.learning_rate = learning_rate  # alpha
        self.decay_rate = decay_rate  # lambda
        self.random_probability = epsilon_max # epsilon
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.mdp = mdp
        self.policy = np.full((len(mdp.states), len(mdp.actions)), 0.25)
        self.v_values = np.zeros((len(mdp.states)))
        self.gamma = 1 #TODO is this ok?
        self.epsilon_seq = [1.0]

    def learn(self, percept, episode_num):
        self.evaluate(percept)
        self.improve(episode_num)

    @abc.abstractmethod
    def evaluate(self, percept):
        pass

    # @abc.abstractmethod
    def improve(self, episode_num):
        for s in range(len(self.mdp.states)):
            action_values = np.zeros(len(self.mdp.actions))
            for a in range(len(self.mdp.actions)):
                action_value = 0
                rsa = self.mdp.get_specific_rsa(s, a)
                for s2 in range(len(self.mdp.states)):
                    action_value += (self.mdp.get_specific_ptsa(s, a, s2) * (rsa + self.gamma * self.v_values[s2]))
                action_values[a] = action_value

            optimal_action = np.random.choice(np.where(action_values == action_values.max())[0])
            # print(optimal_action)

            for a in range(len(self.actions)):
                if optimal_action == a:
                    self.policy[s][a] = 1 - self.random_probability + (self.random_probability / len(self.actions))
                else:
                    self.policy[s][a] = self.random_probability / len(self.actions)

        old_random = self.random_probability
        self.random_probability = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * math.exp((-self.decay_rate) * episode_num)

        if old_random != self.random_probability:
         self.epsilon_seq.append(self.random_probability)
