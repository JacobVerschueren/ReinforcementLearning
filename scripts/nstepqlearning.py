from scripts.learningStrategy import LearningStrategy
import numpy as np


class NstepQlearning(LearningStrategy):
    """
    qlearning subclass of the learning strategy
    """

    def __init__(self, mdp, learning_rate, decay_rate, epsilon_max = 1.0, epsilon_min = 0.01, number_of_steps):
        LearningStrategy.__init__(self, mdp, learning_rate, decay_rate, epsilon_max, epsilon_min)
        self.states = mdp.get_states()
        self.actions = mdp.get_actions()
        self.qvalues = np.zeros((len(self.states), len(self.actions)))
        self.number_of_steps = number_of_steps
        self.percept_list = []

    def evaluate(self, percept):
        self.mdp.update(percept)

        if len(self.percept_list >= self.number_of_steps):
            for i in range(len(self.percept_list)):
                self.qvalues[percept[0], percept[1]] += (self.α * (self.mdp.get_specific_rsa(percept[0], percept[1]) + self.γ * (np.max(self.qvalues[percept[2]] - self.qvalues[percept[0], percept[1]]))))
        # TODO code in for loop not done
        for s in range(len(self.states)):
            self.v_values[s] = np.max(self.qvalues[s])

        if percept[4]:
            self.percept_list = []
