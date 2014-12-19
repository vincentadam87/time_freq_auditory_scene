"""
Testing Shepard and variants elements
"""

import unittest
import inspect


from TimeFreqAuditoryScene import *

class Test_Shepard(unittest.TestCase):
    """
    Testing sound generation for leaf elements
    - ShepardTone
    - Tritone
    - ShepardRisset
    - ShepardFM
    """

    TAG = "Test_Shepard"
    def setUp(self):
        self.fs = 44100.
        self.duration = 0.3


    def test_Shepard(self):
        self.logPoint()
        genv = GaussianSpectralEnvelope(mu=960.,sigma_oct=1.)
        items = [
            ShepardTone(env=genv),
            Tritone(env=genv),
            ShepardRisset(env=genv),
            ShepardFM(env=genv)
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
        self.test_Shepard()


if __name__ == '__main__':
    unittest.main()
