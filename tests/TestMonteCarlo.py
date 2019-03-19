import unittest
from scripts.markovDecisionProcess import MarkovDecisionProcess
from scripts.percept import Percept
from scripts.learning_strategies.monteCarlo import MonteCarlo
import math


class TestMdp(unittest.TestCase):
    """
    run unittests from console with:
    python -m discover ./tests
    """

    def setUp(self):
        self.mdp = MarkovDecisionProcess()
        self.monte = MonteCarlo(self.mdp, 0.8, 0.001, 0.6, 1.0, 0.01)
        self.monte.mdp.reset()
        self.monte.evaluate(Percept(0,2,1,1,False))
        self.monte.evaluate(Percept(0,2,4,2,False))

    def test_rewards(self):
        self.assertEqual(self.monte.mdp.rewards[0, 2, 1], 1)

    def test_nsa(self):
        self.assertEqual(self.monte.mdp.nsa[0, 2], 2)

    def test_ntsa(self):
        self.assertEqual(self.monte.mdp.ntsa[0, 2, 1], 1)

    def test_ptsa(self):
        self.assertEqual(self.monte.mdp.ptsa[0, 2, 1], 0.5)

    def test_epsilon(self):
        self.monte.improve(1)
        self.assertEqual(self.monte.ε, self.monte.epsilon_min +
                         (self.monte.epsilon_max - self.monte.epsilon_min) * math.exp((-self.monte.λ)))

if __name__ == '__main__':
    unittest.main()
