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
                percept = Percept(self.state, action, new_state, reward, final_state)
                print('percept: ', percept)
                self.strategy.learn(percept, episode_count)
                self.state = percept.new_state
                print('State after update: ', self.state, '\n')
                episode_done = percept.final_state

            self.state = self.environment.reset()
            episode_count += 1


        # TODO betere visualisatie
        # actions = ['left ', 'down ', 'right', ' up  ']
        print_array = np.zeros((4,4))
        for i in range(4):
            for j in range (4):
                print_array[i,j] = np.argmax(self.strategy.policy[i*4+j])
        print(print_array)
        #  print(self.strategy.mdp.rewards[14])
        #  print(self.strategy.mdp.rsa[14])
