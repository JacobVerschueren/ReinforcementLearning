from scripts.learning_strategies.learningStrategy import LearningStrategy
from scripts.percept import Percept
from scripts.utils.makeGraph import MakeGraph
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
                action = np.random.choice(
                    self.strategy.mdp.actions,
                    1,
                    p=self.strategy.policy[self.state])[0]

                new_state, reward, final_state, unnecessary_prob = self.environment.step(action)
                percept = Percept(self.state, action, new_state, reward, final_state)
                self.strategy.learn(percept)
                self.state = percept.new_state
                episode_done = percept.final_state

            self.state = self.environment.reset()
            self.strategy.episode_count += 1
            episode_count = self.strategy.episode_count

            # updating the visualization list
            max_policy = np.argmax(self.strategy.policy, axis=1)
            for i in range(self.strategy.n_states):
                self.visualisationList[i] = self.fl_actions[max_policy[i]]

            """
            if self.strategy.episode_count%1000 == 0:
                print(np.reshape(self.strategy.v_values, (4, -1)))
                print(self.strategy.qvalues)
            """
            if self.strategy.episode_count % 100 == 0:
                graph = MakeGraph("Average reward over 100 episodes", self.strategy.percentage_rewardlist, "reward over last 100")
                graph.drawGraph()
