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

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        sd = SceneDrawer(ax)

        c = [{"line":[0,2,100,300]},
             {"box":[0,1,100,200]},
             {"function":{"handle":lambda x:100*x**2, "type":"frequency",
                         "start":0.7,
                         "duration":1}}]
        sd.drawCommand(c)
        plt.show()

    def logPoint(self):
        currentTest = self.id().split('.')[-1]
        callingFunction = inspect.stack()[1][3]
        print 'in %s - %s()' % (currentTest, callingFunction)

    def runTest(self):
        print "\nRunning " + self.TAG
        self.test_all()


if __name__ == '__main__':
    unittest.main()
