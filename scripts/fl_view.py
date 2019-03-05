import kivy
from kivy.properties import ObjectProperty
kivy.require('1.10.1')
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.widget import Widget
from kivy.clock import Clock
from scripts.agent import Agent
from scripts.markovDecisionProcess import MarkovDecisionProcess
from scripts.qlearning import Qlearning


class FlGame(GridLayout):
    mdp = MarkovDecisionProcess(0, range(16), range(4))
    strategy = Qlearning(mdp, 1, 0.1, 1.0, 0.01)
    algorithm = Agent(strategy, 'FrozenLake-v0')
    algorithm.learn(10)
    testText = 1

    state1 = ObjectProperty(None)

    def update(self, dt):
        self.label1.text = str(self.testText)
        self.testText += 1


class FlApp(App):

    def build(self):
        fl = FlGame()
        #Clock.schedule_interval(fl.update, 10.0 / 60.0)
        return fl


if __name__ == '__main__':
    FlApp().run()