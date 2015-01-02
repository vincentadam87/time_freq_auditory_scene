"""
Testing some Node elements
"""

import unittest
import inspect


from TimeFreqAuditoryScene import *

class Test_Nodes(unittest.TestCase):
    """
    Testing sound generation for nodes elements
    """

    TAG = "Test_Nodes"
    def setUp(self):
        self.fs = 44100.
        self.duration = 0.3


    def test_Nodes(self):
        self.logPoint()
        genv = GaussianSpectralEnvelope(mu=960.,sigma_oct=1.)
        items = [ToneSequence(intertone_delay=0.1,
                              tone_duration=0.5,
                              freqs=[100.,200.,100.],
                              env=genv),
                 UniformToneSequence(intertone_delay=0.1,
                                     tone_duration=0.5,
                                     n_tones=5,
                                     band=[100.,200.],
                                     env=genv),
                 ConstantIntervalChord(fb=100.,
                                       interval=2.,
                                       duration=0.2,
                                       env=genv)]

        for item in items:
            x = item.generate(self.fs)
            print(item.TAG)
            self.assertIsInstance(x, np.ndarray)
            self.assertTrue(x.ndim == 1)


    def logPoint(self):
        currentTest = self.id().split('.')[-1]
        callingFunction = inspect.stack()[1][3]
        print 'in %s - %s()' % (currentTest, callingFunction)

    def runTest(self):
        print "\nRunning " + self.TAG
        self.test_Nodes()


if __name__ == '__main__':
    unittest.main()
