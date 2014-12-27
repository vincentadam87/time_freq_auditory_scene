"""
Testing Leaf elements
"""

import unittest
import inspect

from pylab import *

from TimeFreqAuditoryScene import *

class Test_Leaf(unittest.TestCase):
    """
    Testing sound generation for leaf elements
    - Tone
    - AMTone
    - Sweep
    - InstantaneousFrequency
    - WhiteNoise
    """

    TAG = "Test_Leaf"
    def setUp(self):
        self.fs = 44100.
        self.duration = 0.3


    def test_Leaf(self):
        self.logPoint()
        leaves = [
            Tone(freq=200., duration=self.duration),
            AMTone(freq=200., duration=self.duration, fmod=10.),
            Sweep(freqs=[100.,200.], duration=self.duration),
            InstantaneousFrequency(phase=lambda t: np.exp(t)),
            WhiteNoise(duration=self.duration)
            ]

        for leaf in leaves:
            print(leaf.TAG)
            x = leaf.generate(self.fs)
            self.assertIsInstance(x, np.ndarray)
            self.assertTrue(x.ndim == 1)


    def logPoint(self):
        currentTest = self.id().split('.')[-1]
        callingFunction = inspect.stack()[1][3]
        print 'in %s - %s()' % (currentTest, callingFunction)

    def runTest(self):
        print "\nRunning " + self.TAG
        self.test_Leaf()


if __name__ == '__main__':
    unittest.main()
