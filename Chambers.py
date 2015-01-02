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
                    type="chords",  # chords or streams
                    bias='up',
                    List=[], delay=0., scale=1.):

        super(Context, self).__init__(delay=delay, List=List, scale=scale)

        self.n_tones = n_tones
        self.tone_duration = tone_duration
        self.inter_tone_interval = inter_tone_interval
        self.env = env
        self.fb_T1 = fb_T1
        self.bias = bias
        self.type = type

        assert fb_T1 is not None
        bias_sign = 1. if self.bias == 'up' else -1.

        assert type in ["chords", "streams"]
        List = []
        shifts = 2.**(bias_sign*np.random.rand(self.n_tones,)*6./12.)
        fmin = 10.
        fmax = 44000.

        # Construction as a (horizontal) sequence of consecutive Shepard Tones (chords)
        if type == "chords":
            runTime = 0
            for i in range(self.n_tones):
                st = ShepardTone(fb=shifts[i]*fb_T1,
                                 duration=self.tone_duration,
                                 delay=runTime,
                                 env=self.env,
                                 fmin=fmin,
                                 fmax=fmax)
                List.append(st)
                runTime += self.tone_duration+self.inter_tone_interval

        # Construction as a (vertical) stack of simultaneous tone Sequences (streams)
        elif type == "streams":
            imin = int(1./np.log(2)*np.log(fmin/fb_T1))
            imax = int(1./np.log(2)*np.log(fmax/fb_T1))
            indices = np.arange(imin, imax)

            for i in indices:
                fb = 2**i*fb_T1
                ts = ToneSequence(intertone_delay=inter_tone_interval,
                            tone_duration=tone_duration,
                            freqs=[shift*fb for shift in shifts],
                            env=env)
                List.append(ts)

        self.List = List








#class RandomDropOutContext(Node):
#    """
#    A sequence of shepard Tones
#    For each shepard tone, a fixed number of pure tones is removed
#    """


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