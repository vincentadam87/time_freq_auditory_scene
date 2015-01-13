"""
Testing Leaf elements
"""

import unittest
import inspect

from matplotlib import pyplot as plt

from TimeFreqAuditoryScene import *

class Test_SceneDrawer(unittest.TestCase):
    """
    Testing sceneDrawer
    """

    TAG = "Test_Leaf"
    def setUp(self):
        self.fs_plot = 500.


    def test_all(self):
        self.logPoint()

        sd = SceneDrawer()

        genv = GaussianSpectralEnvelope(mu=960.,sigma_oct=1.)
        sp = ShepardTone(env=genv)

        sd.draw(sp)

    def logPoint(self):
        currentTest = self.id().split('.')[-1]
        callingFunction = inspect.stack()[1][3]
        print 'in %s - %s()' % (currentTest, callingFunction)

    def runTest(self):
        print "\nRunning " + self.TAG
        self.test_all()


if __name__ == '__main__':
    unittest.main()
