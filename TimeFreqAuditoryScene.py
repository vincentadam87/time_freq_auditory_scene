"""

This module provides classes to declare, play and plot auditory scene populated by elements in the (time, frequency) domain
The scene is organized as a tree with Leaves as atom elements of the scene.

The tree offers a way to group atoms, and make group of groups etc...

This is useful:
    - to declare structured groups of atoms (ex. shepard tone)
    - to easily create repeated patterns (ex. time shifted groups)

.. _Google Python Style Guide:
   http://google-styleguide.googlecode.com/svn/trunk/pyguide.html

"""

import numpy as np
from abc import *
from matplotlib import pyplot as plt
import copy
import collections

class Node(object):
    """Node

    Attributes:
        List (list): list of Leaf elements in the node
        delay (float): delay relative to absolute scene start time
        scale (float): scaling to be applied to all elements deeper in tree
        active (bool): flag declaring inclusion/exclusion from scene
        index (int) : additional flag, useful for ordering items
    """
    TAG = "Node"

    def __init__(self, List=[], delay=0., scale=1., active=True, index=None, parent=None, draw_bbox=False):
        self.List=[]
        self.add(List)
        self.delay = delay
        self.scale = scale
        self.active = active
        self.index = index
        self.parent = parent
        self.draw_bbox = draw_bbox

    def getduration(self):
        """Duration of node

        Returns:
            duration of node

        """
        return np.max([item.delay+item.getduration() for item in self.List if item.active==True])

    def getstart(self):
        """Start time relative to root"""
        if self.parent is not None:
            return self.parent.getstart()+self.delay
        else:
            return 0

    def getAbsScale(self):
        """Scale relative to root"""
        if self.parent is not None:
            return self.parent.getAbsScale()*self.scale
        else:
            return 1

    def getTbox(self):
        start = self.getstart()
        return [start,start+self.getduration()]

    def getFbox(self):
        Fbox = [float("inf"), float("-inf")]
        for item in self.List:
            Fbox_item = item.getFbox()
            Fbox[1] = max(Fbox[1],Fbox_item[1])
            Fbox[0] = min(Fbox[0],Fbox_item[0])
        return Fbox

    def getbbox(self):
        """
        Get the smallest time/freq box containing elements of the node

        Returns:
            [tmin, tmax, fmin, fmax] (absolute time and frequency wrt origin of scene)
        """
        return self.getTbox()+self.getFbox()


        #rel_box = [self.getstart(), 0, float("inf"), float("-inf")] # init relative box
        #for item in self.List:
        #    if item.active is True:
        #        # get subnode box
        #        item_box = item.getbbox()
        #        item_box[1]+=self.delay
        #        # update current node box
        #        rel_box[0] = rel_box[0] if rel_box[0] < item_box[0] else item_box[0]
        #        rel_box[2] = rel_box[2] if rel_box[2] < item_box[2] else item_box[2]
        #        rel_box[1] = rel_box[1] if rel_box[1] > item_box[1] else item_box[1]
        #        rel_box[3] = rel_box[3] if rel_box[3] > item_box[3] else item_box[3]

        return rel_box

    def add(self, item):
        """Populate node

        Args:
            item (Node/Leaf or list): element or subtree to add to node
        """
        if type(item) == list:
            for i in item:
                self.add(i)
        else:
            item.parent = self
            self.List.append(item)

    def flatten(self, cum_delay, prod_scale):
        l = []
        for item in self.List:
            l += item.flatten(cum_delay, prod_scale)

    def print_content(self):
        for item in self.List:
            item.print_content()

    def generate(self, fs):
        """Generate soundwave

        Args:
            fs (float): sampling frequency (Hz)

        """
        duration = self.getduration()
        x = np.zeros((int(duration*fs),))
        for item in self.List:
            if item.active is True:
                xt = item.generate(fs)
                i_start = int(item.delay*fs)
                m = min(len(xt)-1, len(x)-i_start-1)
                x[i_start:i_start + m] += xt[0:m]*self.scale
        return x

    def draw(self, ax, prop_delay, prop_scale):
        """Draw node

        Args:
            ax (matplotlib.axes.Axes): axe instance to plot into
            prop_delay (float): propagated absolute delay of parent node
            prop_scale (float): propagated absolute scale of parent node

        """
        for node in self.List:
            if node.active is True:
                node.draw(ax, prop_delay+self.delay, prop_scale*self.scale)

        if self.draw_bbox==True:
            color = "black"
            box = self.getbbox()
            # horizontal
            ax.plot([box[0], box[1]], [box[2], box[2]], color=color)
            ax.plot([box[0], box[1]], [box[3], box[3]], color=color)
            # vertical
            ax.plot([box[0], box[0]], [box[2], box[3]], color=color)
            ax.plot([box[1], box[1]], [box[2], box[3]], color=color)

class Scene(Node):
    """Scene

    Scene, implements Node + has extra method to draw spectrogram
    It is the root reference for time and scale

    """

    TAG = "Scene"

    def __init__(self, List=[]):
        """Constructor"""
        super(Scene, self).__init__(List=List)

    def flatten_scene(self):
        """
        Clear all hierarchy in terms of embedded lists
        children remember who there parent is
        parents have no memory of their children
        """
        self.List = [item for item in flatten(self.List)]

class Leaf(object):
    """Abstract Leaf Class

    Attributes:
        delay (float): delay wrt to start of parent node, default to 0.
        duration (float): duration of leaf.
        active (bool): flag for inclusion/exclusion in scene, default to 0.
        index (int): additional index descriptor, default to None.

    """
    Parent = None
    TAG = "Leaf"

    def __init__(self, delay=0., duration=1.,amp=1., active=True, index=None, parent=None):
        """Constructor"""
        self.delay = delay
        self.duration = duration
        self.amp = amp
        self.active = active
        self.index = index
        self.parent = parent

    def getstart(self):
        """Start time relative to root"""
        if self.parent is not None:
            return self.parent.getstart()+self.delay
        else:
            return 0

    @abstractmethod
    def generate(self, fs):
        raise NotImplementedError()

    @abstractmethod
    def getbbox(self):
        raise NotImplementedError()

    def getduration(self):
        return self.duration

    def getAbsScale(self):
        """Scale relative to root"""
        if self.parent is not None:
            return self.parent.getAbsScale()*self.amp
        else:
            return 1

    @abstractmethod
    def print_content(self):
        raise NotImplementedError()

    @abstractmethod
    def makePlotOpts(self):
        """ Should always return a list of dictionaries """
        raise NotImplementedError()

    @abstractmethod
    def draw(self, ax, prop_delay, prop_scale):
        raise NotImplementedError()

class Tone(Leaf):
    """Tone

    Attributes:
        freq (float): frequency of pure tone.
        amp (float): amplitude of pure tone.
        phase (float): phase of pure tone.
        ramp: (float): duration of cosine ramp
    """

    TAG = "Tone"

    def __init__(self, freq=100., delay=0., duration=1.,  amp=1., ramp=0.01, phase=None, index=None, active=True):
        """Constructor

        Phase is randomly set if none given.
        """
        super(Tone, self).__init__(delay=delay, duration=duration, amp=amp, index=index, active=active)
        self.freq = freq
        self.ramp = ramp
        # random phase if not specified
        if phase is None:
            self.phase = np.random.rand()*np.pi*2.

    def generate(self, fs):
        s = self.getstart()
        d = self.getduration()
        t = np.linspace(s, s+d, int(d*fs))
        if self.freq<fs/2:  # anti-alias safety
            y = np.cos(2.*np.pi*self.freq*t + self.phase)
        else:
            y = np.zeros(t.shape)
        # apply border smoothing in the form of a raising squared cosine amplitude modulation
        Ltau = int(self.ramp*fs)
        up = 1.-np.cos(np.linspace(0, np.pi/2., Ltau))**2
        down = np.cos(np.linspace(0, np.pi/2., Ltau))**2
        y[0:Ltau] *= up
        y[-Ltau:] *= down
        return y*self.amp

    def getduration(self):
        return self.duration

    def getbbox(self):
        return [self.delay, self.delay+self.duration, self.freq, self.freq]

    def getTbox(self):
        return [self.delay, self.delay+self.duration]

    def getFbox(self):
        return [self.freq, self.freq]

    def print_content(self):
        print(self.TAG+\
              ", freq:"+str(self.freq)+\
              ", amp:"+str(self.amp)+\
              ", duration:"+str(self.duration)+\
              ", delay:"+str(self.delay))

    def draw(self, ax, prop_delay, prop_scale):
        abs_amp = self.amp*prop_scale
        map_abs_amp = sig((abs_amp-0.2))
        ax.plot([prop_delay + self.delay, prop_delay + self.delay + self.duration],
                [self.freq, self.freq],
                lw=map_abs_amp*4.,
                alpha=map_abs_amp,
                color='black')

    def makePlotOpts(self):
        s = self.getstart()
        d = self.getduration()
        return [{"line":
                     {"tf":[s, s+d, self.freq, self.freq],
                      "a":self.getAbsScale()}
                }]

class SAMTone(Tone):
    """Sinusoid Amplitude Modulated Tone

    Attributes:
        fmod (float): frequency of envelope modulator

    """

    TAG = "SAMTone"

    def __init__(self, freq=100., delay=0., duration=1.,  amp=1., fmod=10.):
        """Constructor"""
        super(SAMTone, self).__init__(freq=freq,
                                      amp=amp,
                                      delay=delay,
                                      duration=duration)
        self.fmod = fmod

    def generate(self, fs):
        t = np.linspace(0., self.duration, int(self.duration*fs))
        y = np.cos(2.*np.pi*self.freq*t)*(0.5+0.5*np.cos(2.*np.pi*self.fmod*t))
        # apply border smoothing in the form of a raising squared cosine amplitude modulation
        tau = 0.02  # 10ms
        Ltau = int(tau*fs)
        up = 1.-np.cos(np.linspace(0, np.pi/2., Ltau))**2
        down = np.cos(np.linspace(0, np.pi/2., Ltau))**2
        y[0:Ltau] *= up
        y[-Ltau:] *= down
        return y*self.amp

    def getduration(self):
        return self.duration

    def print_content(self):
        print(self.TAG+\
              ", freq:"+str(self.freq)+\
              ", amp:"+str(self.amp)+\
              ", fmod:"+str(self.fmod)+\
              ", duration:"+str(self.duration)+\
              ", delay:"+str(self.delay))

    def draw(self, ax, prop_delay, prop_scale):
        abs_amp = self.amp*prop_scale
        map_abs_amp = sig((abs_amp-0.2))
        tmod = 1./self.fmod
        t = np.arange(0,self.duration,tmod)
        for ti in t:
            #print  ti, ti+0.5*tmod, self.duration
            ax.plot([prop_delay + self.delay + ti, prop_delay + self.delay + ti+0.5*tmod],
                    [self.freq, self.freq],
                    lw=map_abs_amp*4.,
                    alpha=map_abs_amp,
                    color='black')

class Sweep(Leaf):
    """Frequency Sweep

    Linear interpolation in frequency domain

    Attributes
        freqs (list): frequency bounds of the sweep
    """

    TAG = "Sweep"

    def __init__(self, freqs=[100.,200.], delay=0., duration=1.,  amp=1.):

        super(Sweep, self).__init__(amp=amp,
                                    delay=delay,
                                    duration=duration)
        self.freqs = freqs

    def generate(self, fs):
        s = self.getstart()
        d = self.getduration()
        t = np.linspace(s, s+d, int(d*fs))
        f = np.linspace(self.freqs[0], self.freqs[1], len(t))
        y = np.cos(2.*np.pi*f*t)
        # apply border smoothing in the form of a raising squared cosine amplitude modulation
        tau = 0.02  # 10ms
        Ltau = int(tau*fs)
        up = 1.-np.cos(np.linspace(0, np.pi/2., Ltau))**2
        down = np.cos(np.linspace(0, np.pi/2., Ltau))**2
        y[0:Ltau] *= up
        y[-Ltau:] *= down
        return y*self.amp

    def getduration(self):
        return self.duration

    def getbbox(self):
        return [self.delay, self.delay+self.duration, min(self.freqs), max(self.freqs)]

    def getTbox(self):
        return [self.delay, self.delay+self.duration]

    def getFbox(self):
        return [min(self.freqs), max(self.freqs)]

    def print_content(self):
        print(self.TAG+\
              ", freqs:"+str(self.freqs)+\
              ", amp:"+str(self.amp)+\
              ", duration:"+str(self.duration)+\
              ", delay:"+str(self.delay))

    def draw(self, ax, prop_delay, prop_scale):
        abs_amp = self.amp*prop_scale
        map_abs_amp = sig((abs_amp-0.2))
        ax.plot([prop_delay+self.delay, prop_delay+self.delay+ self.duration],
                [self.freqs[0], self.freqs[1]],
                lw=map_abs_amp*4.,
                alpha=map_abs_amp,
                color='black')

    def makePlotOpts(self):
        s = self.getstart()
        d = self.getduration()
        return [{"line":
                     {"tf":[s, s+d, self.freqs[0], self.freqs[1]],
                      "a":self.getAbsScale()}
                }]

class InstantaneousFrequency(Leaf):
    """Instantaneous frequency Leaf
    Sound cos(f(t))
    f is the instantaneous frequency

    Attributes
        phase (function): instantaneous phase.
        i_freq (function): instantaneous frequency
        env: spectral envelope

    """

    TAG = "InstantaneousFrequency"

    def __init__(self, phase=None, i_freq=None, delay=0., duration=1., amp=1.,  env=None):

        super(InstantaneousFrequency, self).__init__(delay=delay,
                                                     duration=duration,
                                                     amp=amp)
        self.phase = copy.deepcopy(phase)
        self.i_freq = copy.deepcopy(i_freq)
        self.env = env


    def generate(self, fs):
        s = self.getstart()
        d = self.getduration()
        t = np.linspace(s, s+d, int(d*fs))-self.delay
        if self.phase is not None:
            phase = self.phase(t)  # the time instantaneous phase
            ift = np.diff(phase)  # the instantaneous frequency
            phase = phase[:-1]
        elif self.i_freq is not None:
            ift = self.i_freq(t)/fs
            phase = np.cumsum(ift)

        if self.env is not None:
            ampt = self.env.amp(ift*fs)  # the instantaneous amplitude
        else:
            ampt = self.amp

        y = np.cos(2.*np.pi*phase)*ampt
        # apply border smoothing in the form of a raising squared cosine amplitude modulation
        tau = 0.02  # 10ms
        Ltau = int(tau*fs)
        up = 1.-np.cos(np.linspace(0, np.pi/2., Ltau))**2
        down = np.cos(np.linspace(0, np.pi/2., Ltau))**2
        y[0:Ltau] *= up
        y[-Ltau:] *= down
        return y  #*self.amp

    def getduration(self):
        return self.duration

    def print_content(self):
        print(self.TAG+\
              ", f:"+str(self.f_phase)+\
              ", amp:"+str(self.amp)+\
              ", duration:"+str(self.duration)+\
              ", delay:"+str(self.delay))

    def draw(self, ax, prop_delay, prop_scale):
        fs_plot = 2000.
        n = int(self.duration*fs_plot)
        t = np.linspace(0., self.duration, n )  # time support
        if self.phase is not None:
            phase = self.phase(t)  # the time instantaneous phase
            ift = np.diff(phase)*fs_plot  # the instantaneous frequency
        elif self.i_freq is not None:
            ift = self.i_freq(t)
            ift = ift[:-1]

        ax.plot(prop_delay+self.delay+t[:-1],ift,
                color='black')

    def makePlotOpts(self):
        s = self.getstart()
        d = self.getduration()
        return [{"function":{
            "start":self.getstart(),
            "delay":self.delay,
            "duration":self.getduration(),
            "handle": self.phase if self.phase is not None else self.i_freq,
            "type": "phase" if self.phase is not None else "frequency"}}]

class SpectralEnvelope(object):
    """Abstract Spectral envelope"""

    def __init__(self, env):
        self.env = env

    @abstractmethod
    def amp(self, x):
        raise NotImplementedError()

class GaussianSpectralEnvelope(SpectralEnvelope):
    """
    Gaussian Spectral envelope

    Attributes:
        mu_log (float): log frequency mean
        sigma_log (float): log frequency std
    """

    def __init__(self, **kwargs):
        """Constructor

        Args:
            **kwargs:
                mu_log (mean in log domain).
                mu (mean).
                sigma_log (sigma in log domain).
                sigma_oct (sigma in octaves).
        """
        if 'mu_log' in kwargs:
            self.mu_log=kwargs['mu_log']
        elif 'mu' in kwargs:
            self.mu_log=np.log(kwargs['mu'])
        if 'sigma_log' in kwargs:
            self.sigma_log = kwargs['sigma_log']
        elif 'sigma_oct' in kwargs:
            self.sigma_log = kwargs['sigma_oct']*np.log(2.)

    def amp(self, x):
        return g_env(x, self.mu_log, self.sigma_log)

class UnitSpectralEnvelope(SpectralEnvelope):
    """
    Unit Spectral envelope
    """
    def __init__(self):
        pass
    def amp(self, x):
        # returns 1
        return np.ones(np.array(x).shape)

class Chord(Node):
    """Chord

    Chord of pure tones

    Attributes:
        freqs (list): List of pure tone frequencies
        amps (list): List of pure tone amplitudes
        duration (float): shared duration of all tones

        fb (float, optional): base frequency, when constructing form intervals
        intervals (list, optional): intervals between tones

    """

    TAG = "Chord"

    def __init__(self, duration=1., delay=0., amps=None, ramp=0.01, env=None, index=None, **kwargs):
        """
        Chord of pure tones constructor
        # Multiple ways of constructing the chord
        # - from frequencies
        # -- and amplitude
        # -- and spectral envelope
        # - from a base freq and intervals
        """

        super(Chord, self).__init__(delay=delay, index=index)
        self.duration = duration
        self.env = env
        self.ramp = ramp

        # constructing from preexisting tone
        if 'chord' in kwargs:
            self.freqs = kwargs['chord'].freqs
            self.amps = kwargs['chord'].amps
            self.add(kwargs['chord'].List)
            self.duration = kwargs['chord'].duration
            self.env = kwargs['chord'].env

        else:
            # Constructing frequencies
            # - constructing from pairs
            if 'freqs' in kwargs:
                self.freqs = kwargs['freqs']
            # - constructing from base and intervals
            elif 'fb' in kwargs:
                assert 'intervals' in kwargs
                self.freqs = kwargs['fb']*np.cumprod(kwargs['intervals'])


            # Constructing amplitudes

            # - from input
            if amps != None:
                self.amps = kwargs['amps']
            # - from enveloppe
            else:
                if env != None:
                    self.env = env
                else:
                    self.env = UnitSpectralEnvelope()

                self.amps = self.env.amp(self.freqs)



            assert len(self.freqs) == len(self.amps)
            self.build_tones()

    def build_tones(self):
        self.List=[]
        n_chord = len(self.freqs)
        for i in range(n_chord):
            tone = Tone(freq=self.freqs[i],
                        delay=0.,
                        duration=self.duration,
                        amp=self.amps[i],
                        ramp=self.ramp,
                        index=i)
            self.add(tone)

    def shift_tones(self, shift=1.):
        """
        Shift all tones,
        either by a common shift if argument is scalar
        or by individual shifts

        Args:
            shift (float or list): scaling of all frequencies of tones in chord

        """
        if isinstance(shift, list):
            assert len(shift) == len(self.freqs)
        self.freqs = (np.asarray(self.freqs)*np.asarray(shift)).tolist()
        self.amps = self.env.amp(self.freqs) # won't work if no enveloppe
        self.build_tones()

class WhiteNoise(Leaf):
    """WhiteNoise
    """

    TAG = "WhiteNoise"

    def __init__(self, delay=0., duration=1.,  amp=1.):

        super(WhiteNoise, self).__init__(delay=delay,
                                         duration=duration,
                                         amp=amp)

    def generate(self, fs):
        t = np.linspace(0., self.duration, int(self.duration*fs))
        y = np.random.randn(len(t))
        # apply border smoothing in the form of a raising squared cosine amplitude modulation
        tau = 0.02  # 10ms
        Ltau = int(tau*fs)
        up = 1.-np.cos(np.linspace(0, np.pi/2., Ltau))**2
        down = np.cos(np.linspace(0, np.pi/2., Ltau))**2
        y[0:Ltau] *= up
        y[-Ltau:] *= down
        return y*self.amp

    def getduration(self):
        return self.duration

    def getbbox(self):
        return [self.delay, self.delay+self.duration, 0, float("+inf")]

    def print_content(self):
        print(self.TAG+\
              ", amp:"+str(self.amp)+\
              ", duration:"+str(self.duration)+\
              ", delay:"+str(self.delay))

    def draw(self, ax, prop_delay, prop_scale):
        abs_amp = self.amp*prop_scale
        ax.plot([prop_delay+self.delay, prop_delay+self.delay+ self.duration],
                [100, 100],color='red')
        ax.plot([prop_delay+self.delay, prop_delay+self.delay+ self.duration],
                [1000, 1000],color='red')

    def makePlotOpts(self):
        s = self.getstart()
        d = self.getduration()
        return [{"box":[s, s+d, 0, float("inf")]}]

class ConstantIntervalChord(Chord):
    """
    ConstantIntervalChord

    Attributes
        fb (float): base frequency
        interval (float): constant frequency interval

    """

    TAG = "ConstantIntervalChord"
    
    def __init__(self, fb=50., interval=2., duration=1., delay=0., ramp=0.01, env=None,index=None, List=[], fmin=5, fmax=40000):
        """ConstantIntervalChord  constructor

        Args
            fb (float): base frequency
            interval (float): constant frequency interval

        """

        imin = int(1./np.log(interval)*np.log(fmin/fb))
        imax = int(1./np.log(interval)*np.log(fmax/fb))
        index = np.arange(imin, imax)

        freqs = []
        for i in index:
            fi = interval**i*fb
            freqs.append(fi)

        super(ConstantIntervalChord, self).__init__(freqs=freqs,
                                                    delay=delay,
                                                  duration=duration,
                                                  ramp=ramp,
                                                  List=List,
                                                  env=env,
                                                  index=index)
        self.fb = fb
        self.fmin = fmin
        self.fmax = fmax

class HarmonicComplexTone(Chord):
    """Harmonic Complex Tone"""

    TAG = "HarmonicComplexTone"

    def __init__(self, f0=100, harmonics=None, duration=1., delay=0., ramp=0.01, env=None, index=None, List=[], fmin=5, fmax=40000 ):

        imax = int(fmax/f0)

        if harmonics is None:
            harmonics = range(1, imax)
        elif isinstance(harmonics, list):
            harmonics = [h for h in harmonics if h<imax]

        freqs = []
        for i in harmonics:
            freqs.append(i*f0)

        super(HarmonicComplexTone, self).__init__(freqs=freqs,
                                                    delay=delay,
                                                  duration=duration,
                                                  ramp=ramp,
                                                  List=List,
                                                  env=env,
                                                  index=index)
        self.f0 = f0
        self.harmonics = harmonics
        self.fmin = fmin
        self.fmax = fmax



class ShepardTone(ConstantIntervalChord):
    """Shepard Tone


    """

    TAG = "ShepardTone"

    def __init__(self, fb=50., duration=1., delay=0., ramp=0.01, env=None,index=None, List=[], fmin=5, fmax=40000):
        """Shepard Tone constructor

        Args:
            fb (float): base frequency
        """

        super(ShepardTone, self).__init__(fb=fb,
                                          interval=2.,
                                          delay=delay,
                                          duration=duration,
                                          List=List,
                                          env=env,
                                          ramp=ramp,
                                          index=index,
                                          fmin=fmin,
                                          fmax=fmax)

class Tritone(Node):
    """
    TriTone (octave interval)

    Attributes:
        fb (float): base frequency of the first tone
        duration_sp (float): duration of shepard tones in tritone
        delay_sp (float): delay between shepard tones in tritone
    """

    TAG = "Tritone"

    def __init__(self, fb=50., duration_sp=1., delay_sp=0., delay=0., ramp=0.01, env=None, fmin=5, fmax=40000):
        """TriTone constructor

        Args:
            fb (float): base frequency of first tone
            duration_sp (float): duration of shepard tones in tritone
            delay_sp (float): delay between shepard tones in tritone
        """
        super(Tritone, self).__init__(delay=delay)
        T1 = ShepardTone(fb=fb, duration=duration_sp, delay=0., ramp=ramp, env=env, fmin=fmin, fmax=fmax,index=0)
        T2 = ShepardTone(fb=fb*np.sqrt(2.), duration=duration_sp, delay=duration_sp+delay_sp, ramp=ramp, env=env, fmin=fmin, fmax=fmax,index=1)
        self.add([T1, T2])
        self.fb = fb
        self.duration_sp = duration_sp
        self.delay_sp = delay_sp

class ShepardRisset(Node):
    """ShepardRisset Tone

    Attributes:
        k (float): the directed speed factor of the base frequency
    """

    TAG = "ShepardRisset"

    def __init__(self, fb=50., interval=2., duration=1., delay=0., env=None, List=[], k=1.1, **kwargs):
        """ShepardRisset constructor

        Args:
            fb (float): base frequency
            duration (float): duration
            k (float): the directed speed factor of the base frequency
        """

        super(ShepardRisset, self).__init__(delay=delay, List=List)
        self.k =k
        self.interval = interval
        # backward construction from ending base frequency
        if 'fb_end' in kwargs:
            self.fb = kwargs['fb_end']*np.exp(-k*duration)
        else:
            self.fb = fb


        fmin = 5.
        fmax = 40000.
        imin = np.ceil(1./np.log(interval)*np.log(fmin/self.fb))
        imax = np.floor(1./np.log(interval)*np.log(fmax/self.fb))
        index = np.arange(imin, imax)
        self.List = []


        # fixed form of temporal evolution to have constant speed over the circle
        def phase(fi):
            return lambda x: fi*np.exp(x*k)/k

        i_restart = -1 if k < 0 else 0
        f_thresh = fmin if k < 0 else fmax

        # initial tones have duration set corresponding to when they cross fmax
        for i in index:
            fi = interval**i*self.fb
            duration_tone = np.abs(1./k*np.log(f_thresh/fi))
            instFreq = InstantaneousFrequency(phase=phase(fi),
                                              duration=min(duration_tone,duration),
                                              env=env)
            self.add(instFreq)

        # added tones appearing as times goes on
        dt = np.abs(1./k*np.log(interval))
        times = np.arange(dt,duration,dt)

        for time in times:
            fi = interval**index[i_restart]*self.fb
            duration_tone = np.abs(1./k*np.log(f_thresh/fi))
            instFreq = InstantaneousFrequency(phase=phase(fi),
                                              delay=time,
                                              duration=min(duration-time,duration_tone),
                                              env=env)
            self.add(instFreq)

class ShepardFM(Node):
    """ShepardFM

    Shepard Tone with frequency modulated base frequency (no new incoming tones)
    """

    TAG = "ShepardFM"


    def __init__(self, fb=50., interval=2., duration=1., delay=0., env=None, List=[], amod=0.25, fmod=10., phase=0., **kwargs):
        """Shepard Tone FM constructor

        Args:
            fb: base frequency
            interval: the interval between tones (2. for shepard)
            delay:
            amod: amplitude of the frequency modulation in terms of octave
            fmod: frequency of modulation
            duration: duration
            env: function (log f -> amplitude)
        """
        super(ShepardFM, self).__init__(delay=delay, List=List)
        self.interval = interval
        self.amod = amod
        self.fmod = fmod
        self.fb = fb

        fmin = 5.
        fmax = 40000.
        imin = np.ceil(1./np.log(interval)*np.log(fmin/self.fb))
        imax = np.floor(1./np.log(interval)*np.log(fmax/self.fb))
        index = np.arange(imin, imax)
        self.List = []


        def inst_freq(fi):
            return lambda t: fi*np.exp(np.log(interval)*amod*np.cos(2*np.pi*fmod*t+phase))

        for i in index:
            fi = interval**i*self.fb
            instFreq = InstantaneousFrequency(i_freq=inst_freq(fi), duration=duration, env=env)
            self.add(instFreq)

class ToneSequence(Node):
    """Tone Sequence

    Sequence of tones of same duration, same inter-tone delay
    """
    TAG = "ToneSequence"

    def __init__(self, intertone_delay=0.1,
                 tone_duration=0.5,
                 freqs=None,
                 ramp=0.01,
                 List=[], delay=0., scale=1., env=None):
        """Constructor

        Args:
            freqs (list): list of tone frequencies
        """
        super(ToneSequence, self).__init__(delay=delay, List=List, scale=scale)
        self.intertone_delay=intertone_delay
        self.tone_duration=tone_duration
        self.freqs=freqs
        self.env=env
        self.ramp=ramp
        assert isinstance(self.freqs, list)
        self.build_tones()

    def build_tones(self):
        self.List=[]
        runTime = 0
        index = 0
        for f in self.freqs:
            amp = self.env.amp(f) if self.env is not None else 1.
            tone = Tone(freq=f, duration=self.tone_duration, delay=runTime, amp=amp, ramp=self.ramp, index=index)
            self.add(tone)
            runTime += self.tone_duration + self.intertone_delay
            index += 1

    def shift_tones(self, shift=1.):
        """
        Shift all tones,
        either by a common shift if argument is scalar
        or by individual shifts

        Args:
            shift (float or list): scaling of all frequencies of tones in chord

        """
        self.freqs = (np.asarray(self.freqs)*np.asarray(shift)).tolist()
        self.amps = self.env.amp(self.freqs) # won't work if no enveloppe
        self.build_tones()

class UniformToneSequence(ToneSequence):
    """Uniform Tone Sequence

    Sequence of tones of same duration, same inter-tone delay
    Frequencies independently log_uniformly drawn from within a frequency band

    Attributes:
        band: band from which tone frequencies are sampled

    """
    TAG = "UniformToneSequence"

    def __init__(self, intertone_delay=0.1,
                 tone_duration=0.5,
                 band=[100,200],
                 n_tones=5,
                 ramp=0.01,
                 List=[], delay=0., scale=1., env=None):
        """Constructor

        Args:
            band (list): band from which tone frequencies are sampled
            n_tones (int): number of tones in sequence

        """

        freqs = list( band[0]* (band[1]/band[0])**np.random.rand(n_tones,)  )

        super(UniformToneSequence, self).__init__(
                intertone_delay=intertone_delay,
                 tone_duration=tone_duration,
                 freqs=freqs,
                 ramp=ramp,
                 List=List, delay=delay, scale=scale, env=env)
        self.band=band
        self.freqs = freqs

# ----------------------

class SceneDrawer(object):

    def __init__(self, f_axis="log", map=None):
        self.f_axis = f_axis
        if map is None:
            self.map = lambda x:sig((x-0.2))

    def draw(self, item, ax=None):

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.set_yscale(self.f_axis)
            ax.set_ylabel(self.f_axis=="log" and "log freq (Hz)" or "freq (Hz)")
            ax.set_xlabel("time (s)")
            ax.set_title("Spectrogram")

        """ Draw spectrogram for item in tree (recursive) """
        if isinstance(item, Leaf):
            if item.active is True:
                self.drawLeaf(item, ax)
        elif isinstance(item, Node):
            if item.active is True:
                for sub_item in item.List:
                    self.draw(sub_item, ax=ax)

    def makePlotOpts(self, item):
        """ Make plot instructions from node item (recursive) """
        if isinstance(item, Node):
            plotOpts = []
            for sub_item in item.List:
                plotOpts += self.makePlotOpts(sub_item)
            return plotOpts
        if isinstance(item, Leaf):
            return item.makePlotOpts()

    def drawLeaf(self, leaf, ax):
        """ Parsing the different plotOptions for a leaf"""
        plotOpts = leaf.makePlotOpts()
        plotOpts = self.applyGlobalOpts(plotOpts)
        for item in plotOpts:
            self.drawCommand(item, ax)

    def applyGlobalOpts(self, plotOpts):
        """ if global options, add them"""
        return plotOpts

    def drawCommand(self, k, ax):
        """parsing the drawing commands"""

        if type(k) is list:
            for item in k:
                self.drawCommand(item, ax)

        elif type(k) is dict:

            if "line" in k:
                # iso amplitude line
                v = k["line"]
                tf = v["tf"]
                a = v["a"]
                ax.plot([tf[0], tf[1]],[tf[2],tf[3]], lw=1+map(a), alpha=1.0, color=str(1.-a))

            if "box" in k:
                v = k["box"]
                ax.plot([v[0], v[1]],[v[2],v[2]], lw=1,alpha=1, color='black')
                ax.plot([v[0], v[1]],[v[3],v[3]], lw=1,alpha=1, color='black')
                ax.plot([v[0], v[0]],[v[2],v[3]], lw=1,alpha=1, color='black')
                ax.plot([v[1], v[1]],[v[2],v[3]], lw=1,alpha=1, color='black')

            if "function" in k:
                fs_plot = 2000.

                v = k["function"]
                s = v["start"]
                d = v["duration"]
                fn = v["handle"]
                de = v["delay"]
                n = int(d*fs_plot)
                t = np.linspace(s, s+d, n)  # time support


                assert v["type"] in ["phase","frequency"]
                if v["type"] == "phase":
                    phase = fn(t-de)  # the time instantaneous phase
                    ift = np.diff(phase)*fs_plot  # the instantaneous frequency
                elif v["type"] == "frequency":
                    ift = fn(t-de)
                    ift = ift[:-1]

                ax.plot(t[:-1],ift, color='black')

# ----------------------

def g_env(x, mu_log, sigma_log):
    """Unnormalized gaussian envelope

    Args:
        x: where to evaluate
        mu: mean
        sigma: standard deviation
    """
    return np.exp(-(np.log(x)-mu_log)**2/sigma_log**2)

def sig(x):
    """Standard sigmoid function"""
    return 1./(1+np.exp(-x))

def map(x):
    return sig(10*(x-0.2))

def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, basestring):
            for sub in flatten(el):
                yield sub
        else:
            yield el

# ----------------------

if __name__ == "__main__":
    pass
