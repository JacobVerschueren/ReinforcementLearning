import numpy as np


class MarkovDecisionProcess:
    """
    This class implements the Markov Decision Process which will serve as the environment in the Learning Algorithm
    """

    def __init__(self,
                 state=0,
                 states=range(16),
                 actions=range(4),
                 #rewards=np.zeros((16, 4, 16)),
                 #nsa=np.zeros((16, 4)),
                 #ntsa=np.zeros((16, 4, 16)),
                 #ptsa=np.zeros((16, 4, 16))
                 ):
        self.state = state
        self.states = states
        self.actions = actions
        self.rewards = np.zeros((len(states), len(actions), len(states)))
        self.nsa = np.zeros((len(states), len(actions)))
        self.ntsa = np.zeros((len(states), len(actions), len(states)))
        self.ptsa = np.zeros((len(states), len(actions), len(states)))
        self.rsa = np.zeros((len(states), len(actions))) # average reward associated with a specific action in a specific state

    def calculate_ptsa(self):
        """
        calculates ptsa from nta and ntsa
        """
        for s1 in range(len(self.ntsa)):
            for a in range(len(self.ntsa[s1])):
                for s2 in range(len(self.ntsa[s1, a])):
                    if self.ntsa[s1, a, s2] == 0:
                        self.ptsa[s1, a, s2] = 0
                    else:
                        self.ptsa[s1, a, s2] = self.ntsa[s1, a, s2]/self.nsa[s1, a]

    def update(self, percept):
        old_state, action, new_state, reward, final_state = percept
        self.rewards[old_state, action, new_state] = reward
        self.nsa[old_state, action] += 1
        self.ntsa[old_state, action, new_state] += 1
        #print(self.ntsa[old_state, action, new_state])
        self.update_ptsa(old_state, action)
        self.update_rsa(old_state, action)
        if final_state:
            self.state = 0
        else:
            self.state = new_state

    def update_ptsa(self, old_state, action):
        for s2 in range(len(self.ntsa[old_state][action])):
            if self.ntsa[old_state][action][s2] == 0:
                self.ptsa[old_state][action][s2] = 0
            else:
                self.ptsa[old_state][action][s2] = self.ntsa[old_state][action][s2] / self.nsa[old_state][action]

    def update_rsa(self, old_state, action):
        reward = 0
        for s2 in range(len(self.ptsa[old_state, action])):
            reward += self.ptsa[old_state, action, s2] * self.rewards[old_state, action, s2]

    def reset(self):
        self.state = 0
        self.rewards = np.zeros((len(self.states), len(self.actions), len(self.states)))
        self.nsa = np.zeros((len(self.states), len(self.actions)))
        self.ntsa = np.zeros((len(self.states), len(self.actions), len(self.states)))
        self.ptsa = np.zeros((len(self.states), len(self.actions), len(self.states)))
        self.rsa = np.zeros((len(self.states), len(self.actions)))

    def get_states(self):
        return self.states

    def get_actions(self):
        return self.actions

    def get_specific_rsa(self, state, action):
        return self.rsa[state, action]

    def get_specific_ptsa(self, old_state, action, new_state):
        return self.ptsa[old_state, action, new_state]
