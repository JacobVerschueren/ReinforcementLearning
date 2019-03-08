from scripts.learningStrategy import LearningStrategy
from scripts.percept import Percept
import gym
import numpy as np


class Agent:
    """
    Algorithm that performs actions (dictated by the learning strategy) in the environment
    and passes the information gained back to the learning strategy
    """

    def __init__(self, strategy: LearningStrategy, environment):
        self.environment = gym.make(environment)
        self.state = self.environment.reset()
        self.strategy = strategy
        self.visualisationList = np.chararray(self.strategy.n_states, unicode=True)
        self.visualisationList[:] = '\u2190'
        self.fl_actions = ['\u2190', '\u2193', '\u2192', '\u2191']  # arrow characters for the frozen lake game

    def learn(self, n_episodes):
        episode_count = 0
        while episode_count < n_episodes:
            episode_done = False
            while not episode_done:
                # print('old_state: ', self.state)
                # print(self.strategy.policy[self.state])
                action = np.random.choice(
                    self.strategy.mdp.actions,
                    1,
                    p=self.strategy.policy[self.state])[0]
                # print('Action: ', action)
                new_state, reward, final_state, unnecessary_prob = self.environment.step(action)
                # print('new_state: ', new_state)
                percept = Percept(self.state, action, new_state, reward, final_state)
                # print('percept: ', percept)
                self.strategy.learn(percept) # TODO if view test doesn't work add count
                self.state = percept.new_state
                # print('State after update: ', self.state, '\n')
                episode_done = percept.final_state

            self.state = self.environment.reset()
            episode_count += 1
            # updating the visualization list
            max_policy = np.argmax(self.strategy.policy, axis=1)
            for i in range(self.strategy.n_states):
                self.visualisationList[i] = self.fl_actions[max_policy[i]]
            # print(self.visualisationList)
