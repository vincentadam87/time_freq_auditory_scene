"""
Testing Chambers Sound Classes
"""

import unittest
import inspect
import random

from Chambers import *
from TimeFreqAuditoryScene import *

class Test_Chambers(unittest.TestCase):
    """
    Testing sound generation for leaf elements
    - Context
    - Clearing
    """

    TAG = "Test_Chambers"
    def setUp(self):
        self.fs = 44100.
        self.tone_duration = 0.3
        self.n_tones = 5
        self.inter_tone_interval = 0.1

    def test_Chambers(self):
        self.logPoint()
        genv = GaussianSpectralEnvelope(mu=960.,sigma_oct=1.)
        bias = random.choice(['up','down'])
        items = [
            Context(n_tones=self.n_tones, inter_tone_interval=self.inter_tone_interval, env=genv, bias=bias, fb_T1=1.),
            Clearing(n_tones=self.n_tones, inter_tone_interval=self.inter_tone_interval, env=genv)
            ]

        for item in items:
            x = item.generate(self.fs)
            self.assertIsInstance(x, np.ndarray)
            self.assertTrue(x.ndim == 1)

    def logPoint(self):
        currentTest = self.id().split('.')[-1]
        callingFunction = inspect.stack()[1][3]
        print 'in %s - %s()' % (currentTest, callingFunction)

    def runTest(self):
        print "\nRunning " + self.TAG
        self.test_Chambers()


if __name__ == '__main__':
    unittest.main()
