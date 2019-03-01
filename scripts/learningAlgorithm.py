from scripts.learningStrategy import LearningStrategy
from scripts.markovDecisionProcess import MarkovDecisionProcess
import gym
import random
import numpy as np

class LearningAlgorithm:

    def __init__(self, strategy: LearningStrategy, environment):
        self.environment = gym.make(environment)
        self.state = self.environment.reset()
        self.strategy = strategy

    def learn(self, n_episodes):
        episode_count = 0
        while episode_count < n_episodes:
            episode_done = False
            while not episode_done:
                print('old_state: ', self.state)
                print(self.strategy.policy[self.state])
                action = np.random.choice(
                    self.strategy.mdp.actions,
                    1,
                    p=self.strategy.policy[self.state])[0]
                print('Action: ', action)
                new_state, reward, final_state, unnecessary_prob = self.environment.step(action)
                print('new_state: ', new_state)
                percept = [self.state, action, new_state, reward, final_state]
                print('percept: ', percept)
                self.strategy.learn(percept, episode_count)
                self.state = percept[2]
                print('State after update: ', self.state, '\n')
                episode_done = percept[4]

            self.state = self.environment.reset()
            episode_count += 1


        # TODO betere visualisatie
        # actions = ['left ', 'down ', 'right', ' up  ']
        print_array = np.zeros((4,4))
        for i in range(4):
            for j in range (4):
                print_array[i,j] = np.argmax(self.strategy.policy[i*4+j])
        print(print_array)
