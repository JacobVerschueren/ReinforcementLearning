import cProfile
from scripts.markovDecisionProcess import MarkovDecisionProcess
from scripts.percept import Percept
from scripts.learning_strategies.valueIteration import ValueIteration


mdp = MarkovDecisionProcess(0, range(16), range(4))
valueTest = ValueIteration(mdp, 0.8, 0.001, 0.90, 0.9, 1.0, 0.01)
percept = Percept(0,2,4,2,False)

cProfile.run('for i in range(1000):'
             '  valueTest.evaluate(percept)')
