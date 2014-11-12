"""
Classes and functions to declare, play and plot auditory scene populated by elements in the (time, frequency) domain
The scene is organized as a tree with Leaves as atom elements of the scene.
The tree offers a way to group atoms, and make group of groups etc...
This is useful:
- to declare structured groups of atoms (ex shepard tone)
- to easily create repeated patterns (time shifted groups)
"""

import numpy as np
from abc import *
from scipy.io.wavfile import write
from sound4python import sound4python as S4P
import os
from matplotlib import pyplot as plt
import copy

class Node(object):
    """
    Node
    - scale and delay are applied to children
    """
    TAG = "Node"
    List = None
    delay = None
    scale = None

    def __init__(self, List=[], delay=0., scale=1.):
        self.List = List
        self.delay = delay
        self.scale = scale


    def getduration(self):
        return np.max([item.delay+item.getduration() for item in self.List])

    def addItemToList(self, item):
        if type(item) == list:
            self.List = self.List + item
        else:
            self.List.append(item)

    def print_content(self):
        for item in self.List:
            item.print_content()

    def generate(self, fs):
        duration = self.getduration()
        x = np.zeros((int(duration*fs),))
        for item in self.List:
            xt = item.generate(fs)
            i_start = int(item.delay*fs)
            x[i_start:i_start+len(xt)] += xt*self.scale
        return x

    def draw(self, ax, prop_delay, prop_scale):
        """
        Draw node, function recursively calls children in tree down to leaves
        #:param ax: pyplot ax
        #:param prop_delay: cumulative delay from root
        #:param prop_delay: multiplicated scaling from root
        """
        for node in self.List:
            node.draw(ax, prop_delay+self.delay, prop_scale*self.scale)

class Scene(Node):
    """
    Scene, implements Node + has extra method to draw spectrogram
    It is the root reference for time and scale
    """

    def __init__(self, List=[]):

        super(Scene, self).__init__()
        self.List = []
        self.TAG = "Scene"


    def draw_spectrogram(self, f_axis="log"):
        """
        Returns a matplotlib figure containing drawn spectrogram
        """
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for node in self.List:
            node.draw(ax, prop_delay=0., prop_scale=1.)
        ax.set_yscale(f_axis)
        ax.set_ylabel(f_axis=="log" and "log freq (Hz)" or "freq (Hz)")
        ax.set_xlabel("time (s)")
        ax.set_title("Spectrogram")
        return fig

class Leaf(object):
    """
    Abstract Leaf
    """

    TAG = "Leaf"
    delay = None
    duration = None

    def __init__(self, delay=0., duration=1.):
        self.delay = delay
        self.duration = duration

    @abstractmethod
    def generate(self, fs):
        raise NotImplementedError()

    def getduration(self):
        return self.duration

    @abstractmethod
    def print_content(self):
        raise NotImplementedError()

    @abstractmethod
    def draw(self, ax, prop_delay, prop_scale):
        raise NotImplementedError()

class Tone(Leaf):
    """
    Tone
    """

    TAG = "Tone"

    def __init__(self, freq=100., delay=0., duration=1.,  amp=1.):

        super(Tone, self).__init__(delay=delay, duration=duration)
        self.freq = freq
        self.amp = amp

    def generate(self, fs):
        t = np.linspace(0., self.duration, int(self.duration*fs))
        y = np.cos(2.*np.pi*self.freq*t)
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
        print self.TAG+\
              ", freq:"+str(self.freq)+\
              ", amp:"+str(self.amp)+\
              ", duration:"+str(self.duration)+\
              ", delay:"+str(self.delay)

    def draw(self, ax, prop_delay, prop_scale):
        abs_amp = self.amp*prop_scale
        map_abs_amp = sig((abs_amp-0.2))
        ax.plot([prop_delay, prop_delay+ self.duration],
                [self.freq, self.freq],
                lw=map_abs_amp*4.,
                alpha=map_abs_amp,
                color='black')

class Sweep(Leaf):
    """
    Frequency Sweep (Linear interpolation in frequency domain)
    """

    TAG = "Sweep"

    def __init__(self, freqs=[100.,200.], delay=0., duration=1.,  amp=1.):

        super(Sweep, self).__init__(delay=delay, duration=duration)
        self.freqs = freqs
        self.amp = amp

    def generate(self, fs):
        n = int(self.duration*fs)
        t = np.linspace(0., self.duration, n )
        f = np.linspace(self.freqs[0], self.freqs[1], n)
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

    def print_content(self):
        print self.TAG+\
              ", freqs:"+str(self.freqs)+\
              ", amp:"+str(self.amp)+\
              ", duration:"+str(self.duration)+\
              ", delay:"+str(self.delay)

    def draw(self, ax, prop_delay, prop_scale):
        abs_amp = self.amp*prop_scale
        map_abs_amp = sig((abs_amp-0.2))
        ax.plot([prop_delay+self.delay, prop_delay+self.delay+ self.duration],
                [self.freqs[0], self.freqs[1]],
                lw=map_abs_amp*4.,
                alpha=map_abs_amp,
                color='black')

class InstantaneousFrequency(Leaf):
    """
    Sound cos(f(t))
    f is the instantaneous frequency
    """

    TAG = "InstantaneousFrequency"

    def __init__(self, f=None, delay=0., duration=1.,  amp=1.):

        super(InstantaneousFrequency, self).__init__(delay=delay, duration=duration)
        self.f = copy.deepcopy(f)
        self.amp = amp

    def generate(self, fs):
        n = int(self.duration*fs)
        t = np.linspace(0., self.duration, n )  # time support
        ft = self.f(t)  # the time instantaneous frequency
        y = np.cos(2.*np.pi*ft)

        # apply border smoothing in the form of a raising squared cosine amplitude modulation
        tau = 0.02  # 10ms
        Ltau = int(tau*fs)
        up = 1.-np.cos(np.linspace(0, np.pi/2., Ltau))**2
        down = np.cos(np.linspace(0, np.pi/2., Ltau))**2
        y[0:Ltau] *= up
        y[-Ltau:] *= down
        return y #*self.amp

    def getduration(self):
        return self.duration

    def print_content(self):
        print self.TAG+\
              ", f:"+str(self.f)+\
              ", amp:"+str(self.amp)+\
              ", duration:"+str(self.duration)+\
              ", delay:"+str(self.delay)

    def draw(self, ax, prop_delay, prop_scale):
        fs_plot = 2000.
        n = int(self.duration*fs_plot)
        t = np.linspace(0., self.duration, n )  # time support
        ft = self.f(t)  # the time instantaneous frequency
        dftdt = np.diff(ft)*fs_plot
        abs_amp = self.amp*prop_scale
        map_abs_amp = sig((abs_amp-0.2))
        ax.plot(t[:-1],dftdt,
                color='black')

class SpectralEnvelope(object):
    """
    Abstract Spectral envelope
    """

    def __init__(self, env):
        self.env = env

    @abstractmethod
    def amp(self, x):
        raise NotImplementedError()

class GaussianSpectralEnvelope(SpectralEnvelope):
    """
    Gaussian Spectral envelope
    """
    mu_log = None
    sigma_log = None

    def __init__(self, mu_log=2., sigma_log=1.):
        self.mu_log = mu_log
        self.sigma_log = sigma_log

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
    """
    Static Chord of pure tones
    """

    def __init__(self, duration=1., delay=0., amps=None, env=None, **kwargs):
    #freqs=[], amps=[] List=[]):
        """
        Chord of pure tones constructor
        # Multiple ways of constructing the chord
        # - from frequencies
        # -- and amplitude
        # -- and spectral envelope
        # - from a base freq and intervals
        """

        super(Chord, self).__init__(delay=delay)
        self.List = []
        self.duration = duration
        self.TAG = "Chord"
        self.env = None

        # constructing from preexisting tone
        if 'chord' in kwargs:
            self.freqs = kwargs['chord'].freqs
            self.amps = kwargs['chord'].amps
            self.List = kwargs['chord'].List
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
                        amp=self.amps[i])
            self.List.append(tone)

    def shift_tones(self, shift=1.):
        """
        Shift all tones,
        either by a common shift if argument is scalar
        or by individual shifts
        """
        if isinstance(shift, list):
            assert len(shift) == len(self.freqs)
        self.freqs = (np.asarray(self.freqs)*np.asarray(shift)).tolist()
        self.amps = self.env.amp(self.freqs) # won't work if no enveloppe
        self.build_tones()


class ShepardTone(Chord):
    """
    Shepard Tone
    """

    def __init__(self, fb=50., duration=1., delay=0., env=None, List=[]):
        """
        Shepard Tone constructor
        :param fb: base frequency
        :param duration: duration
        :param env: function (log f -> amplitude)
        """
        self.TAG = "ShepardTone"
        self.fb = fb

        fmin = 10.
        fmax = 20000.
        imin = int(1./np.log(2)*np.log(fmin/fb))
        imax = int(1./np.log(2)*np.log(fmax/fb))
        index = np.arange(imin, imax)
        self.List = []
        self.freqs = []
        self.amps = []
        for i in index:
            fi = 2**i*self.fb
            #tone = Tone(freq=fi, delay=0., duration=duration,  amp=env.amp(fi))
            #self.List.append(tone)
            self.freqs.append(fi)
            self.amps.append(env.amp(fi))

        super(ShepardTone, self).__init__(delay=delay,
                                          duration=duration,
                                          List=List,
                                          env=env)


class ShepardRisset(Node):
    """
    ShepardRisset Tone
    """

    def __init__(self, fb=50., duration=1., delay=0., env=None, List=[], k=1.1):
        """
        Shepard Tone constructor
        :param fb: base frequency
        :param duration: duration
        :param env: function (log f -> amplitude)
        """
        super(ShepardRisset, self).__init__(delay=delay, List=List)
        self.TAG = "ShepardTone"
        self.fb = fb
        self.k =k

        fmin = 5.
        fmax = 20000.
        imin = int(1./np.log(2)*np.log(fmin/fb))
        imax = int(1./np.log(2)*np.log(fmax/fb))
        index = np.arange(imin, imax)
        self.List = []

        def inst_f(fi):
            return lambda x: fi*np.exp(x*k)

        for i in index:
            fi = 2.**i*self.fb
            instFreq = InstantaneousFrequency(f=inst_f(fi), duration=duration,  amp=env.amp(fi))
            self.List.append(instFreq)




# ----------------------

def g_env(x, mu, sigma):
    """
    Unnormalized gaussian envelope
    :param x:
    :param mu: mean
    :param sigma: standard deviation
    :return:
    """
    return np.exp(-(np.log(x)-mu)**2/sigma**2)

def playsound(x, fs=44100.):
    """
    Playing sound x
    :param x:
    :return:
    """
    scaled = np.int16(x/np.max(np.abs(x)) * 32767)
    wavPath = 'tmp.wav'
    if os.path.isfile(wavPath):
        os.remove(wavPath)
    write(wavPath, fs, scaled)
    s4p = S4P.Sound4Python()
    s4p.loadWav(wavPath)
    s4p.play()

def sig(x):
    """
    Standard sigmoid function
    :param x:
    :return:
    """
    return 1./(1+np.exp(-x))

# ----------------------

if __name__ == "__main__":
    print "starting!"
    duration = 0.1
    scene = Scene()
    stList = []

    #------------------------------------------------

    # declaring interval for sequence of shepard tones
    k = 2.**(2./12.)
    # declare gaussian envelope on log frequency
    genv = GaussianSpectralEnvelope(mu_log=5., sigma_log=2.)
    # Constructing the scene
    for i in range(20):
        tmp_st = ShepardTone(fb=10.*(k)**i, env=genv, delay=duration*i, duration=duration)
        stList.append(tmp_st)
    scene.List = stList

    # generate sound
    #x = scene.generate(fs=44100.)
    #playsound(x, fs=44100. )

    # draw spectrogram
    #fig = scene.draw_spectrogram()
    #plt.show()

    #------------------------------------------------

    #scene2 = Scene()
    #factor = 2.**(2./12.)
    #duration = 0.2
    #dt = duration/4.
    #for i in range(20):
    #    start = np.random.rand()
    #    f_start = 200.+500.*np.random.rand()
    #    t_start = np.random.rand()*4.
    #    sweep = Sweep(freqs=[f_start, f_start*factor], delay=dt*i, duration=duration)
    #    scene2.List.append(sweep)


    #x2 = scene2.generate(fs=44100.)
    #playsound(x2)
    #fig2 = scene2.draw_spectrogram()
    #plt.show()

    #------------------------------------------------

    scene3 = Scene()
    duration = 0.3
    fc = 300.
    fmod = 100.
    amod = 1.
    #f = lambda x: fc+amod*np.cos(2*np.pi*fmod*x)
    f = lambda x: 0*x+fc

    vibrato = InstantaneousFrequency(f=f, duration=duration)
    tone = Tone(freq=fc,  duration=duration)
    scene3.List.append(vibrato)
    x3 = scene3.generate(fs=44100.)
    playsound(x3)


    #fig3 = scene3.draw_spectrogram()
    #plt.show()
