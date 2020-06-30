import warnings
from pathlib import Path
from typing import Callable

from IPython.display import Audio
import mimetypes
import torchaudio
from fastai.vision import Image, listify, TfmCrop, FlowField, ItemBase, ifnone
import numpy as np
import math
import torch
import src.loudnorm
import noisereduce as nr
import librosa.display

from .config import AudioConfig
from src.utils import _channel_first, _signal_first

AUDIO_EXTENSIONS = tuple(str.lower(k) for k, v in mimetypes.types_map.items() if v.startswith('audio/'))


class AudioItem(ItemBase):
    def __init__(self, sig=None, sr=None, path=None, spectro=None, max_to_pad=None, start=None, end=None, loudness=None, config=None):
        """Holds Audio signal and/or spectrogram data"""
        if isinstance(sig, np.ndarray): sig = torch.from_numpy(sig)
        self._sig, self._sr, self.path, self.spectro = sig, sr, path, spectro
        self._loudness = loudness
        self.config = config
        self.max_to_pad = max_to_pad
        self.start, self.end = start, end

    def calc(self, func, kwargs):
        if func is Callable:
            return func(self, **kwargs)
        elif isinstance(func, str):
            return getattr(self, func)(**kwargs)

    # @classmethod
    # def open(cls, path:Path):
    #     sig, sr = torchaudio.load(path)
    #     this = cls(sig, sr, path)
    #     return this

    def validate_consistencies(self, config):
        if (config._sr is not None) and (self.sr != config._sr):
            raise ValueError(f'''Multiple sample rates detected. Sample rate {self.sr} of file {self.path} 
                                does not match config sample rate {config._sr} 
                                this means your dataset has multiple different sample rates, 
                                please choose one and set resample_to to that value''')
        if (config._nchannels is not None) and (config._nchannels != self.nchannels):
            raise ValueError(f'''Multiple channel sizes detected. Channel size {self.nchannels} of file 
                                {self.path} does not match others' channel size of {config._nchannels}. A dataset may
                                not contain different number of channels. Please set downmix=true in AudioConfig or 
                                separate files with different number of channels.''')

    def register_spectro(self):
        if self.path is None: raise ValueError("item path wasn't provided")
        cache_path = self.config.get_cache_path(self.path)
        if cache_path.exists():
            spectro = torch.load(cache_path)
        else:
            self.validate_consistencies(self.config)
            spectro = self.create_spectro(self.config)
            if self.config.cache:
                self.config.save_in_cache(self.path, spectro)
        self.spectro = spectro

    def create_spectro(self, config):
        if config.mfcc:
            spec = config.sg_funcs.mfcc(self.sig)
        else:
            spec = config.sg_funcs.spec(self.sig)
            spec = config.sg_funcs.to_mel(spec)
            if config.sg_cfg.to_db_scale:
                spec = config.sg_funcs.to_db(spec)
        spec = spec.detach()
        if config.standardize:
            raise NotImplementedError
            spec = standardize(spec)
        if config.delta:
            raise NotImplementedError
            spec = torch.cat([torch.stack([m,torchdelta(m),torchdelta(m, order=2)]) for m in spec])
        return spec

    def __repr__(self):
        return f'{self.__class__.__name__} {round(self.duration, 2)} s ({self.nchannels} ch, {self.loudness:.2f} LUFS, {self.nsamples} samples @ {self.sr} Hz)'

    def __len__(self): return self.data.shape[0]

    def _repr_html_(self):
        librosa.display.waveplot(self.sig.squeeze().numpy(), sr=self.sr)
        return f'{self.__repr__()}<br />{self.ipy_audio._repr_html_()}'

    def clone(self): return AudioItem(spectro=self.spectro, path=self.path, sr=self.sr, sig=self.sig,
                                      loudness=self.loudness, config=self.config)

    def reconstruct(self, t): return AudioItem(spectro=t)

    def show(self, title: [str] = None, **kwargs):
        print(f"File: {self.path}")
        print(f"Total Length: {round(self.duration, 2)} seconds")
        print(f"Number of Channels: {self.nchannels}")
        images_per_channel = len(self.get_spec_images())/self.nchannels
        self.hear(title=title)
        for i,im in enumerate(self.get_spec_images()):
            print(f"Channel {int(i//images_per_channel)}.{int(i%images_per_channel)} ({im.shape[-2]}x{im.shape[-1]}):")
            display(im.rotate(180).flip_lr())

    def hear(self, title=None):
        if title is not None: print("Label:", title)
        if self.sig is None: self._check_signal()
        if self.start is not None or self.end is not None:
            print(f"{round(self.start/self.sr, 2)}s-{round(self.end/self.sr,2)}s of original clip")
            start = 0 if self.start is None else self.start
            end = self.nsamples-1 if self.end is None else self.end
            display(Audio(data=self.sig[:,start:end], rate=self.sr))
        else:
            display(self.ipy_audio)

    def get_spec_images(self):
        sg = self.spectro
        if sg is None: return []
        return [Image(s.unsqueeze(0)) for s in sg]

    def _preprocess(self):
        """Apply raw waveform preprocessing: loudnorm and noise reduction"""
        # noise removing
        if self.config.silence_threshold:
            self._reduce_noise(self.config.silence_threshold)
        # loudness correcting part
        if self.config.target_loudness: self._set_loudness(self.config.target_loudness, clipping_method='soft_smart')
        return self

    def _reduce_noise(self, silence_threshold: int = 30):
        """a function that cuts the leading and trailing noise from audio signal and uses it as a sample for noise
        reducing algorithm. Original waveform is being replaced. *db* is used as the difference between signal and noise"""
        new_sig, (trim_left, trim_right) = librosa.effects.trim(self.sig, top_db=silence_threshold)
        noise = torch.cat((self.sig[:, :trim_left], self.sig[:, trim_right:]), dim=1)

        if noise.numel():
            # noise is not empty, otherwise cannot perform analysis
            sig = self.sig.squeeze().numpy()
            noise = noise.squeeze().numpy()
            reduced_noise = nr.reduce_noise(audio_clip=sig, noise_clip=noise, verbose=False, prop_decrease=0.9)
            # reshape sig
            self.sig = torch.from_numpy(reduced_noise).view(1, -1)

    def _evaluate_loudness(self, signal, sr):
        if (signal is not None) and (sr is not None):
            loudness = src.loudnorm.get_loudness(signal, sr)
        else:
            loudness = None
        return loudness

    def _set_loudness(self, target_loudness, clipping_method:str = 'soft_smart', **kwargs):
        sig = _channel_first(self.sig)
        input_loudness = self.loudness
        output = src.loudnorm.volume(sig, input_loudness, target_loudness)
        output = src.loudnorm.clip(clipping_method, output, **kwargs)
        self.sig = _signal_first(output)
        if (diff:=abs(self.loudness-target_loudness))>0.5:
            warnings.warn(f'Target loudness not reached due to clipping, {diff=:.2f} LUFS')
            # recurrent execution until the output loudness is within acceptable tolerance
            #self.set_loudness(target_loudness, clipping_method=clipping_method, **kwargs)
        return self

    def _get_resize_target(self, size):
        c, features, time_bins = self.data.shape
        if isinstance(size, int):
            size = (features, size)
        return size

    def _get_duration_crop_target(self, duration):
        *_, features, time_bins = self.data.shape
        # this works for both modes: fourier bins and waveform
        if hasattr(self, 'hl'):
            hl = self.hl
        else:
            hl = math.ceil(self.nsamples / time_bins)
        bins = round(duration/1000 * self.sr / hl)
        return features, bins

    def resize(self, size, interp_mode="bilinear"):
        """Temporary fix to allow image resizing transform"""
        data = self.data.unsqueeze(0)
        size_target = self._get_resize_target(size)
        align_corners = None if interp_mode=='nearest' else False
        data_new = torch.nn.functional.interpolate(data, size=size_target, mode=interp_mode, align_corners=align_corners)
        self.data = data_new.squeeze(0)
        self.sr = self.sr*data_new.shape[-1]/data.shape[-1]

    def apply_tfms(self, tfms, duration:int=1280, size=None, do_resolve:bool=True, padding_mode:str='reflection'):
        tfms = listify(tfms)
        size_tfms = [o for o in tfms if isinstance(o.tfm, TfmCrop)]
        if do_resolve:
            for tfm in tfms:
                tfm.resolve()
        x = self.clone()
        for tfm in tfms:
            # Reset this attribute, otherwise it treats our object as Image instance looking for px and flow fields
            # setattr(tfm.tfm, '_wrap', None)
            if tfm in size_tfms:
                # setattr(tfm.tfm, '_wrap', None)
                crop_target = x._get_duration_crop_target(duration)
                x = tfm(x, size=crop_target, padding_mode=padding_mode)
            else:
                x = tfm(x)
        if size is not None:
            sz = size.get(getattr(self.__class__, '__name__'), x.shape[-1])
            x.resize(sz)
        return x

    def pixel(self, func, **kwargs):
        self.data = func(self.data, **kwargs)
        return self

    def coord(self, func, **kwargs):
        flow = FlowField((1,1), self.data)
        self.data = func(flow, **kwargs).flow
        return self

    def lighting(self, func, **kwargs):
        raise NotImplementedError
        #TODO study audio normalization
        data_new = func(self.logit_px, **kwargs).sigmoid()
        return self

    def save(self, output:Path=''):
        default_name = 'out'
        default_suffix = '.wav'
        output = Path(output)
        if not output.is_absolute():
            if not output.name: output = Path(default_name)
            if not output.suffix: output = output.with_suffix(default_suffix)
            if self.path is not None:
                output = self.path.with_name(output.as_posix())
            else:
                output = Path.cwd().with_name(output.as_posix())
        torchaudio.save(output.as_posix(), src=self.sig, sample_rate=self.sr)

    @property
    def loudness(self):
        if self._loudness is None:
            self._loudness = self._evaluate_loudness(self.sig, self.sr)
        return self._loudness

    @property
    def sig(self):
        if self._sig is None:
            self._load_signal()
        return self._sig

    @sig.setter
    def sig(self, sig):
        self._sig = sig
        self._loudness = self._evaluate_loudness(self.sig, self.sr)

    def _load_signal(self):
        # raise RuntimeError("Shouldn't be reloading signal, what is the purpose?")
        self._sig, self._sr = torchaudio.load(self.path)
        self._preprocess()

    @property
    def sr(self):
        if self._sr is None:
            self._load_signal()
        return self._sr

    @sr.setter
    def sr(self, sr): self._sr = sr

    @property
    def data(self):
        return self.spectro if self.spectro is not None else self.sig

    @data.setter
    def data(self, x):
        if self.spectro is not None:
            self.spectro = x
        else:
            self.sig = x

    @property
    def shape(self): return self.data.shape

    @property
    def ipy_audio(self): 
        if self.sig is None: self._check_signal()
        return Audio(data=self.sig.squeeze().numpy(), rate=self.sr)

    @property
    def duration(self): 
        if(self.sig is not None): return self.nsamples/self.sr
        else: 
            si, ei = torchaudio.info(str(self.path))
            return si.length/si.rate
    
    @property
    def nsamples(self):
        return self.sig.shape[-1]

    @property
    def nchannels(self):
        return self.sig.shape[-2]

