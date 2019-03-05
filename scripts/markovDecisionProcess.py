import numpy as np
from scripts.percept import Percept


class MarkovDecisionProcess:
    """
    This class implements the Markov Decision Process which will serve as the environment in the Learning Algorithm
    """

    def __init__(self,
                 state=0,
                 states=range(16),
                 actions=range(4),
                 ):
        self.state = state
        self.states = states
        self.n_states = len(self.states)
        self.actions = actions
        self.n_actions = len(self.actions)
        self.rewards = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.nsa = np.zeros((self.n_states, self.n_actions))
        self.ntsa = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.ptsa = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.rsa = np.zeros((self.n_states,self.n_actions)) # average reward associated with a specific action in a specific state


    def calculate_ptsa(self):
        """
        calculates ptsa from nta and ntsa
        """
        for s1 in range(self.n_states):
            for a in range(self.n_actions):
                for s2 in range(self.n_states):
                    if self.ntsa[s1, a, s2] == 0:
                        self.ptsa[s1, a, s2] = 0
                    else:
                        self.ptsa[s1, a, s2] = self.ntsa[s1, a, s2]/self.nsa[s1, a]

    def update(self, p: Percept):
        self.nsa[p.old_state, p.action] += 1
        self.ntsa[p.old_state, p.action, p.new_state] += 1
        n = self.ntsa[p.old_state, p.action, p.new_state]
        r = self.rewards[p.old_state, p.action, p.new_state]
        self.rewards[p.old_state, p.action, p.new_state] = np.average([r, p.reward], weights=[n-1/n, 1/n])
        #print(self.ntsa[old_state, action, new_state])
        self.update_ptsa(p.old_state, p.action)
        self.update_rsa(p.old_state, p.action)
        if p.final_state:
            self.state = 0
        else:
            self.state = p.new_state

    def update_ptsa(self, old_state, action):
        for s2 in range(self.n_states):
            if self.ntsa[old_state][action][s2] == 0:
                self.ptsa[old_state][action][s2] = 0
            else:
                self.ptsa[old_state][action][s2] = self.ntsa[old_state][action][s2] / self.nsa[old_state][action]

    def update_rsa(self, old_state, action):
        reward = 0
        for s2 in range(self.n_states):
            reward += self.ptsa[old_state, action, s2] * self.rewards[old_state, action, s2]

        self.rsa[old_state, action] = reward

    def reset(self):
        self.state = 0
        self.rewards = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.nsa = np.zeros((self.n_states, self.n_actions))
        self.ntsa = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.ptsa = np.zeros((self.n_states, self.n_actions, self.n_states))
        self.rsa = np.zeros((self.n_states, self.n_actions))

    def get_states(self):
        return self.states

    def get_actions(self):
        return self.actions

    def get_specific_rsa(self, state, action):
        return self.rsa[state, action]

    def get_specific_ptsa(self, old_state, action, new_state):
        return self.ptsa[old_state, action, new_state]
