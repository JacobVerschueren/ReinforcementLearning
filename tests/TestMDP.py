import unittest
from scripts.markovDecisionProcess import MarkovDecisionProcess


class TestMdp(unittest.TestCase):

    def setUp(self):
        self.mdp = MarkovDecisionProcess()
        self.mdp.reset()
        self.mdp.update([0,2,1,1,False])
        self.mdp.update([0,2,4,2,False])

    def test_rewards(self):
        self.assertEqual(self.mdp.rewards[0, 2, 1], 1)

    def test_nsa(self):
        self.assertEqual(self.mdp.nsa[0, 2], 2)

    def test_ntsa(self):
        self.assertEqual(self.mdp.ntsa[0, 2, 1], 1)

    def test_ptsa(self):
        self.assertEqual(self.mdp.ptsa[0, 2, 1], 0.5)

if __name__ == '__main__':
    unittest.main()
