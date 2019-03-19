import unittest
from scripts.markovDecisionProcess import MarkovDecisionProcess
from scripts.percept import Percept
from scripts.learning_strategies.qlearning import Qlearning
import math


class TestMdp(unittest.TestCase):
    """
    run unittests from console with:
    python -m discover ./tests
    """

    def setUp(self):
        self.mdp = MarkovDecisionProcess()
        self.qlearning = Qlearning(self.mdp, 0.8, 0.01, 0.01, 1.0, 0.01)
        self.qlearning.mdp.reset()
        self.qlearning.evaluate(Percept(0, 2, 1, 1, False))
        self.qlearning.evaluate(Percept(0, 2, 4, 2, False))

    def test_rewards(self):
        self.assertEqual(self.qlearning.mdp.rewards[0, 2, 1], 1)

    def test_nsa(self):
        self.assertEqual(self.qlearning.mdp.nsa[0, 2], 2)

    def test_ntsa(self):
        self.assertEqual(self.qlearning.mdp.ntsa[0, 2, 1], 1)

    def test_ptsa(self):
        self.assertEqual(self.qlearning.mdp.ptsa[0, 2, 1], 0.5)

    def test_epsilon(self):
        self.qlearning.improve(1)
        self.assertEqual(self.qlearning.ε, self.qlearning.epsilon_min +
                         (self.qlearning.epsilon_max - self.qlearning.epsilon_min) * math.exp((-self.qlearning.λ)))

if __name__ == '__main__':
    unittest.main()
