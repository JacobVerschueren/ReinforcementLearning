import unittest
from scripts.markovDecisionProcess import MarkovDecisionProcess
from scripts.percept import Percept
from scripts.learning_strategies.valueIteration import ValueIteration
import math


class TestMdp(unittest.TestCase):
    """
    run unittests from console with:
    python -m discover ./tests
    """

    def setUp(self):
        self.mdp = MarkovDecisionProcess()
        self.value = ValueIteration(self.mdp, 0.8, 0.001, 0.90, 0.9, 1.0, 0.01)
        self.value.mdp.reset()
        self.value.evaluate(Percept(0,2,1,1,False))
        self.value.evaluate(Percept(0,2,4,2,False))

    def test_rewards(self):
        self.assertEqual(self.value.mdp.rewards[0, 2, 1], 1)

    def test_nsa(self):
        self.assertEqual(self.value.mdp.nsa[0, 2], 2)

    def test_ntsa(self):
        self.assertEqual(self.value.mdp.ntsa[0, 2, 1], 1)

    def test_ptsa(self):
        self.assertEqual(self.value.mdp.ptsa[0, 2, 1], 0.5)

    def test_epsilon(self):
        self.value.improve(1)
        self.assertEqual(self.value.ε, self.value.epsilon_min +
                         (self.value.epsilon_max - self.value.epsilon_min) * math.exp((-self.value.λ)))


if __name__ == '__main__':
    unittest.main()
