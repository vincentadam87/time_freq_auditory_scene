"""
Classes to design the sounds of Chambers and Pressnitzer 2014
"""


from TimeFreqAuditoryScene import *
import numpy as np

class Context(Node):

    """
    Context
    A sequence of shepard tones with base frequency sampled uniformly with half an octave up or down of a given base frequency
    """
    TAG = "Context"

    def __init__(self, n_tones=5,
                    tone_duration=0.2,
                    inter_tone_interval=0.1,
                    env=None,
                    fb_T1=None,
                    bias='up',
                    List=[], delay=0., scale=1.):

        super(Context, self).__init__(delay=delay, List=List, scale=scale)

        self.n_tones = n_tones
        self.tone_duration = tone_duration
        self.inter_tone_interval = inter_tone_interval
        self.env = env
        self.fb_T1 = fb_T1
        self.bias = bias

        assert fb_T1 is not None
        bias_sign = 1. if self.bias == 'up' else -1.

        List = []
        runTime = 0
        for i in range(self.n_tones):
            fb = self.fb_T1*2.**(bias_sign*np.random.rand()*6.)
            st = ShepardTone(fb=fb, duration=self.tone_duration, delay=runTime, env=self.env)
            List.append(st)
            runTime += self.tone_duration+self.inter_tone_interval
        self.List = List

class Clearing(Node):
    """
    Clearing stimulus
    A sequence of random base frequency Half-Octave interval shepard tones
    The aim is to wipe out "traces" of previously heard shepard Tones
    """
    TAG = "Clearing"

    def __init__(self, n_tones=5,
                    tone_duration=0.2,
                    inter_tone_interval=0.1,
                    env=None,
                    List = [], delay=0., scale=1.):

        super(Clearing, self).__init__(delay=delay, List=List, scale=scale)
        self.n_tones = n_tones
        self.tone_duration = tone_duration
        self.inter_tone_interval = inter_tone_interval
        self.env = env

        List = []
        runTime = 0
        for i in range(self.n_tones):
            fb = 1.+np.random.rand()
            st = ConstantIntervalChord(fb=fb, interval=np.sqrt(2), duration=self.tone_duration, delay=runTime, env=self.env)
            List.append(st)
            runTime += self.tone_duration+self.inter_tone_interval
        self.List = List