import kivy
from kivy.properties import ObjectProperty
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.clock import Clock
from scripts.agent import Agent
from scripts.markovDecisionProcess import MarkovDecisionProcess
from scripts.qlearning import Qlearning
from scripts.nstepqlearning import NstepQlearning
from scripts.monteCarlo import MonteCarlo
from scripts.valueIteration import ValueIteration
kivy.require('1.10.1')


class FlGame(GridLayout):
    """
    grid layout visualisation of Frozen Lake game
    the learning algorithm is initialised within so make adjustments here
    """
    n_episodes = 50000
    mdp = MarkovDecisionProcess(0, range(16), range(4))
    # strategy = Qlearning(mdp, 0.8, 0.01, 1.0, 0.01)
    # strategy = NstepQlearning(mdp, 0.8, 0.01, 4, 1.0, 0.01)
    # strategy = MonteCarlo(mdp, 0.8, 0.01, 1.0, 0.01)
    strategy = NstepQlearning(mdp, 0.8, 0.01, 0.95, 1.0, 0.01)
    algorithm = Agent(strategy, 'FrozenLake-v0')
    state0 = ObjectProperty(None)
    state1 = ObjectProperty(None)
    state2 = ObjectProperty(None)
    state3 = ObjectProperty(None)
    state4 = ObjectProperty(None)
    state5 = ObjectProperty(None)
    state6 = ObjectProperty(None)
    state7 = ObjectProperty(None)
    state8 = ObjectProperty(None)
    state9 = ObjectProperty(None)
    state10 = ObjectProperty(None)
    state11 = ObjectProperty(None)
    state12 = ObjectProperty(None)
    state13 = ObjectProperty(None)
    state14 = ObjectProperty(None)
    state15 = ObjectProperty(None)

    def update(self, dt):
        if self.algorithm.strategy.episode_count < self.n_episodes:
            self.algorithm.learn(1)
        # print(self.algorithm.strategy.episode_count)
        actions = self.algorithm.visualisationList
        self.state0.text = actions[0]
        self.state1.text = actions[1]
        self.state2.text = actions[2]
        self.state3.text = actions[3]
        self.state4.text = actions[4]
        self.state6.text = actions[6]
        self.state8.text = actions[8]
        self.state9.text = actions[9]
        self.state10.text = actions[10]
        self.state13.text = actions[13]
        self.state14.text = actions[14]


class FlApp(App):
    """
    run this to start
    """

    def build(self):
        fl = FlGame()
        Clock.schedule_interval(fl.update, 5.0 / 60.0)
        return fl


if __name__ == '__main__':
    FlApp().run()
