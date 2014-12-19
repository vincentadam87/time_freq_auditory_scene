"""
Testing Scene construction
"""

import unittest
import inspect

import numpy as np
from TimeFreqAuditoryScene import Scene,Tone

class Test_Scene(unittest.TestCase):
    """
    Testing simple scene construction and soundwave generation
    """

    TAG = "Test_Leaf"
    def setUp(self):
        self.fs = 44100.



    def test_Scene(self):
        self.logPoint()
        leaves = [
            Tone(freq=200., duration=0.1),
            Tone(freq=100., duration=0.1, delay=1.)
            ]
        scene = Scene()
        scene.List = leaves
        x = scene.generate(self.fs)
        self.assertIsInstance(x, np.ndarray)
        self.assertTrue(x.ndim == 1)



    def logPoint(self):
        currentTest = self.id().split('.')[-1]
        callingFunction = inspect.stack()[1][3]
        print 'in %s - %s()' % (currentTest, callingFunction)

    def runTest(self):
        print "\nRunning " + self.TAG
        self.test_Scene()


if __name__ == '__main__':
    unittest.main()
