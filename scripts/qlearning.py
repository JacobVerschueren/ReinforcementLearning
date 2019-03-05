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
        '''
        for s in range(len(self.states)):
            for a in range(len(self.actions)):
        
                self.qvalues[s, a] += (self.learning_rate * (self.mdp.get_specific_rsa(s, a) +
                                                            self.gamma * (np.max(self.qvalues) - self .qvalues[s, a])))
        '''
        '''
        self.qvalues[percept[0], percept[1]] += (self.learning_rate * (percept[3] +
                                                                       self.gamma * (np.max(self.qvalues[percept[2]] - self.qvalues[percept[0], percept[1]]))))
        '''

        self.qvalues[p.old_state, p.action] += \
            (self.α * (self.mdp.get_specific_rsa(p.old_state, p.action) + self.γ *
                       (np.max(self.qvalues[p.new_state] - self.qvalues[p.old_state, p.action]))))

        # TODO code hierboven nakijken

        for s in range(self.n_states):
            self.v_values[s] = np.max(self.qvalues[s])

    '''
    def improve(self, episode_num):
        for s in range(len(self.states)):
            optimal_action = np.random.choice(np.where(self.qvalues[s]==self.qvalues[s].max())[0])
            # print(optimal_action)

            for a in range(len(self.actions)):
                if optimal_action == a:
                    self.policy[s][a] = 1 - self.random_probability + (self.random_probability/len(self.actions))
                else:
                    self.policy[s][a] = self.random_probability/len(self.actions)

        self.random_probability = self.epsilon_min + (self.epsilon_max - self.epsilon_min) \
                                  * math.exp((-self.decay_rate)*episode_num)
    '''
