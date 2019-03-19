import unittest
from scripts.markovDecisionProcess import MarkovDecisionProcess
from scripts.percept import Percept
from scripts.learning_strategies.nstepqlearning import NstepQlearning
import math


class TestMdp(unittest.TestCase):
    """
    run unittests from console with:
    python -m discover ./tests
    """

    def setUp(self):
        self.mdp = MarkovDecisionProcess()
        self.nstep = NstepQlearning(self.mdp, 0.8, 0.001, 0.7, 5, 1.0, 0.01)
        self.nstep.mdp.reset()
        self.nstep.evaluate(Percept(0,2,1,1,False))
        self.nstep.evaluate(Percept(0,2,4,2,False))

    def test_rewards(self):
        self.assertEqual(self.nstep.mdp.rewards[0, 2, 1], 1)

    def test_nsa(self):
        self.assertEqual(self.nstep.mdp.nsa[0, 2], 2)

    def test_ntsa(self):
        self.assertEqual(self.nstep.mdp.ntsa[0, 2, 1], 1)

    def test_ptsa(self):
        self.assertEqual(self.nstep.mdp.ptsa[0, 2, 1], 0.5)

    def test_epsilon(self):
        self.nstep.improve(1)
        self.assertEqual(self.nstep.ε, self.nstep.epsilon_min +
                         (self.nstep.epsilon_max - self.nstep.epsilon_min) * math.exp((-self.nstep.λ)))

if __name__ == '__main__':
    unittest.main()
