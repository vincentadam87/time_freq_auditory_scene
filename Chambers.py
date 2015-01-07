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


class RandomDropOutContext(Context):
    """
    A sequence of shepard Tones
    For each shepard tone, a fixed number of pure tones is removed
    """

    TAG = "RandomDropOutContext"

    def __init__(self, n_tones=5,
                    n_drop = 2,
                    tone_duration=0.2,
                    inter_tone_interval=0.1,
                    env=None,
                    fb_T1=None,
                    type="chords",  # chords
                    bias='up',
                    List=[], delay=0., scale=1.):

        super(RandomDropOutContext, self).__init__(n_tones=n_tones,
                    tone_duration=tone_duration,
                    inter_tone_interval=inter_tone_interval,
                    env=env,
                    fb_T1=fb_T1,
                    type=type,  # chords or streams
                    bias=bias,
                    delay=delay, List=List, scale=scale)

        self.n_drop = n_drop

        # for each shepard tone
        for i in range(self.n_tones):
            st = self.List[i]
            amps = [t.amp for t in st.List]
            i_drops = []
            # remove desired amount of tones
            for d in range(self.n_drop):
                amps /= sum(amps)
                i_drop = sample_discrete(amps,1)
                i_drops.append(i_drop)
                amps[i_drop] = 0
                st.List[i_drop].active = False


class StructuredDropOutContext(Context):
    """
    A sequence of shepard Tones (build as streams)
    For dominant streams (in terms of amplitude), tones are periodically removed

    Attributes:
        n_drop: number of tones dropped at a given time

    """

    TAG = "RandomDropOutContext"

    def __init__(self, n_tones=5,
                    n_drop = 2,
                    tone_duration=0.2,
                    inter_tone_interval=0.1,
                    env=None,
                    fb_T1=None,
                    type="streams",  # streams
                    bias='up',
                    List=[], delay=0., scale=1.):

        super(StructuredDropOutContext, self).__init__(n_tones=n_tones,
                    tone_duration=tone_duration,
                    inter_tone_interval=inter_tone_interval,
                    env=env,
                    fb_T1=fb_T1,
                    type=type,  # chords or streams
                    bias=bias,
                    delay=delay, List=List, scale=scale)
        self.n_drop = n_drop

        # Separate treatment:
        # n_drop = 1, 2

        # identification of dominant streams (2*n_drop)
        n_streams = len(self.List)
        n_tones = self.n_tones

        i_max = []
        amps = [self.List[i].List[0].amp for i in range(n_streams)]
        for j in range(2*n_drop):
            i_max.append(np.argmax(amps))
            amps[i_max[j]] = 0

        print i_max

        # Drop
        for t in range(n_tones):
            for i in i_max:
                a = ((t+i) % 2) == 0
                print a
                self.List[i].List[t].active = a

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
            fb = 2**np.random.rand()  # log-uniform over an octave
            st = ConstantIntervalChord(fb=fb, interval=np.sqrt(2), duration=self.tone_duration, delay=runTime, env=self.env)
            List.append(st)
            runTime += self.tone_duration+self.inter_tone_interval
        self.List = List


def sample_discrete(probabilities, size):
    bins = np.add.accumulate(probabilities)
    return np.digitize(np.random.random_sample(size), bins)