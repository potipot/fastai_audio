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
import loudnorm
import librosa.display

from utils import _channel_first, _signal_first, _to_numpy

AUDIO_EXTENSIONS = tuple(str.lower(k) for k, v in mimetypes.types_map.items() if v.startswith('audio/'))


class AudioItem(ItemBase):
    def __init__(self, sig=None, sr=None, path=None, spectro=None, max_to_pad=None, start=None, end=None, loudness=None):
        """Holds Audio signal and/or spectrogram data"""
        if isinstance(sig, np.ndarray): sig = torch.from_numpy(sig)
        self._sig, self._sr, self.path, self.spectro = sig, sr, path, spectro
        self._loudness = loudness
        self.max_to_pad = max_to_pad
        self.start, self.end = start, end

    @classmethod
    def create(cls, fn: Path, after_open: Callable = None, target_loudness:float=-23.0):
        sig, sr = torchaudio.load(fn)
        if after_open:
            sig = after_open(sig, sr)
            raise NotImplementedError
        #loudness correcting part
        loudness = loudnorm.get_loudness(sig, sr)
        item = cls(sig=sig, sr=sr, path=fn, loudness=loudness)
        if target_loudness: item.set_loudness(target_loudness, clipping_method='soft_smart')
        return item

    def __repr__(self):
        return f'{self.__class__.__name__} {round(self.duration, 2)} s ({self.nchannels} ch, {self.loudness:.2f} LUFS, {self.nsamples} samples @ {self.sr} Hz)'

    def __len__(self): return self.data.shape[0]

    def _repr_html_(self):
        librosa.display.waveplot(self.sig.squeeze().numpy(), sr=self.sr)
        return f'{self.__repr__()}<br />{self.ipy_audio._repr_html_()}'

    def clone(self): return AudioItem(spectro=self.spectro, path=self.path)

    def reconstruct(self, t): return(AudioItem(spectro=t))

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

    def set_loudness(self, target_loudness, clipping_method:str = 'soft_smart', **kwargs):
        sig = _channel_first(self.sig)
        input_loudness = self.loudness
        output = loudnorm.volume(sig, input_loudness, target_loudness)
        output = loudnorm.clip(clipping_method, output, **kwargs)
        self._loudness = loudnorm.get_loudness(output, self.sr)
        self.sig = _signal_first(output)
        if (diff:=abs(self._loudness-target_loudness))>0.5:
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

    def _reload_signal(self): self._sig,self._sr = torchaudio.load(self.path)

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
        loudness = ifnone(self._loudness, loudnorm.get_loudness(self.sig, self.sr))
        self._loudness = loudness
        return self._loudness

    @property
    def sig(self):
        if self._sig is None: self._reload_signal()
        return self._sig

    @sig.setter
    def sig(self, sig): self._sig = sig

    @property
    def sr(self):
        if self._sr is None: self._reload_signal()
        return self._sr

    @sr.setter
    def sr(self, sr): self._sr=sr

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
    def data(self): return self.spectro if self.spectro is not None else self.sig

    @data.setter
    def data(self, x):
        if self.spectro is not None: self.spectro = x
        else:                        self.sig = x
    
    @property
    def nsamples(self): 
        return self.sig.shape[-1]
    
    @property
    def nchannels(self): 
        return self.sig.shape[-2]

