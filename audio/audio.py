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

import noisereduce as nr
import librosa.display

from .config import AudioConfig
from .loudnorm import get_loudness, volume, clip
from .utils import _channel_first, _signal_first

AUDIO_EXTENSIONS = tuple(str.lower(k) for k, v in mimetypes.types_map.items() if v.startswith('audio/'))


class AudioItem(ItemBase):
    def __init__(self, sig=None, sr=None, path=None, spectro=None, max_to_pad=None, start=None, end=None, loudness=None,
                 config:AudioConfig=None):
        """Holds Audio signal and/or spectrogram data"""
        if isinstance(sig, np.ndarray): sig = torch.from_numpy(sig)
        self._sig, self._sr, self.path, self._spectro = sig, sr, path, spectro
        self._loudness = loudness
        self.config = config
        self.max_to_pad = max_to_pad
        self.start, self.end = start, end
        self.is_preprocessed = False
        self.reconstruct_signal = False

    def calc(self, func, kwargs):
        if func is Callable:
            return func(self, **kwargs)
        elif isinstance(func, str):
            return getattr(self, func)(**kwargs)

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

    def _create_spectro(self):
        if self.config.mfcc:
            spec = self.config.sg_funcs.mfcc(self.sig)
        else:
            spec = self.config.sg_funcs.spec(self.sig)
            spec = self.config.sg_funcs.to_mel(spec)
            if self.config.sg_cfg.to_db_scale:
                spec = self.config.sg_funcs.to_db(spec)
        spec = spec.detach()
        if self.config.standardize:
            raise NotImplementedError
            spec = standardize(spec)
        if self.config.delta:
            raise NotImplementedError
            spec = torch.cat([torch.stack([m,torchdelta(m),torchdelta(m, order=2)]) for m in spec])
        return spec

    def __repr__(self):
        return f'{self.__class__.__name__} {round(self.duration, 2)} s ({self.nchannels} ch, {self.loudness:.2f} LUFS, {self.n_samples} samples @ {self.sr} Hz)'

    def __len__(self): return self.data.shape[0]

    def _repr_html_(self):
        librosa.display.waveplot(self.sig.squeeze().numpy(), sr=self.sr)
        return f'{self.__repr__()}<br />{self.ipy_audio._repr_html_()}'

    def clone(self):
        # the following causes an incomplete spectro to be loaded without sr and signal loaded (just data from cache)
        return AudioItem(spectro=self.spectro, path=self.path, config=self.config)

    def reconstruct(self, t): return AudioItem(spectro=t)

    def show(self, title: [str] = None, reconstruct_signal=True, **kwargs):
        if self.path is not None: print(f"File path: {self.path}")
        self.reconstruct_signal = reconstruct_signal

        print(f"Total Length: {round(self.duration, 2)} seconds")
        print(f"Number of Channels: {self.nchannels}")
        images_per_channel = len(self.get_spec_images())/self.nchannels
        self.hear(title=title)
        for i,im in enumerate(self.get_spec_images()):
            print(f"Channel {int(i//images_per_channel)}.{int(i%images_per_channel)} ({im.shape[-2]}x{im.shape[-1]}):")
            display(im.rotate(180).flip_lr())

    def hear(self, title=None):
        if title is not None: print("Label:", title)
        if not self.reconstruct_signal: return
        if self.start is not None or self.end is not None:
            print(f"{round(self.start/self.sr, 2)}s-{round(self.end/self.sr,2)}s of original clip")
            start = 0 if self.start is None else self.start
            end = self.n_samples - 1 if self.end is None else self.end
            display(Audio(data=self.sig[:,start:end], rate=self.sr))
        else:
            display(self.ipy_audio)

    def get_spec_images(self):
        sg = self.spectro
        if sg is None: return []
        return [Image(s.unsqueeze(0)) for s in sg]

    def _preprocess(self):
        """Apply raw waveform preprocessing: downmixing, resampling, pre-emphasis, noise removing and loudnorm."""
        self.is_preprocessed = True
        if self.config is not None:
            # down mixing
            if self.config.downmix:
                self.sig = torch.mean(self.sig, dim=0, keepdim=True)

            # resampling
            if target_sr := self.config.resample_to:
                resampler = torchaudio.transforms.Resample(self.sr, target_sr)
                self.sig = resampler(self.sig)
                self.sr = target_sr

            # pre-emphasis
            if k := self.config.pre_emphasis_coeff:
                assert(self.sig.ndim == 2)
                kernel = torch.tensor([k, 1, 0]).view(1, 1, 3)
                self.sig = torch.nn.functional.conv1d(self.sig.unsqueeze(0), kernel).squeeze(0)
                assert(self.sig.ndim == 2)

            # noise removing
            if self.config.silence_threshold:
                self._reduce_noise(self.config.silence_threshold)

            # loudness normalization
            if self.config.target_loudness:
                self._set_loudness(self.config.target_loudness, clipping_method='soft_smart')

        return self

    def _reduce_noise(self, silence_threshold: int = 30):
        """a function that cuts the leading and trailing noise from audio signal and uses it as a sample for noise
        reducing algorithm. Original waveform is being replaced. *db* is used as the difference between signal and noise"""
        _, (trim_left, trim_right) = librosa.effects.trim(self.sig.squeeze(), top_db=silence_threshold)
        noise = torch.cat((self.sig[:, :trim_left], self.sig[:, trim_right:]), dim=1)

        if noise.numel() > (n_fft := 1024):
            # noise is not empty, otherwise cannot perform analysis
            sig = self.sig.squeeze().numpy()
            noise = noise.squeeze().numpy()
            reduced_noise = nr.reduce_noise(n_fft=n_fft, win_length=n_fft, audio_clip=sig, noise_clip=noise,
                                            verbose=False, prop_decrease=0.9)
            # reshape sig
            self.sig = torch.from_numpy(reduced_noise.astype(np.float32)).view(1, -1)

    def _evaluate_loudness(self, signal, sr):
        if (signal is not None) and (sr is not None):
            loudness = get_loudness(signal, sr)
        else:
            loudness = None
        return loudness

    def _set_loudness(self, target_loudness, clipping_method:str = 'soft_smart', **kwargs):
        sig = _channel_first(self.sig)
        input_loudness = self.loudness
        output = volume(sig, input_loudness, target_loudness)
        output = clip(clipping_method, output, **kwargs)
        self.sig = _signal_first(output)
        if (diff:=abs(self.loudness-target_loudness))>0.5:
            # warnings.warn(f'Target loudness not reached due to clipping, {diff=:.2f} LUFS')
            # recurrent execution until the output loudness is within acceptable tolerance
            self._set_loudness(target_loudness, clipping_method=clipping_method, **kwargs)
        return self

    def _get_resize_target(self, size):
        c, features, time_bins = self.data.shape
        if isinstance(size, int):
            size = (features, size)
        return size

    def _get_duration_crop_target(self, duration):
        *_, features, time_bins = self.data.shape
        # this is probably broken and works only for spectro, not waveform
        if hasattr(self, 'hl'):
            hl = self.hl
        else:
            hl = math.ceil(self.n_samples / time_bins)

        if duration is not None:
            bins = round(duration / 1000 * self.sr / hl)
        else:
            bins = time_bins
        return features, bins

    def resize(self, size, interp_mode="bilinear"):
        """Temporary fix to allow image resizing transform"""
        data = self.data.unsqueeze(0)
        # if data is integer we need to wrap it in float and remember the original type
        source_dtype = data.dtype
        if convert_types := not data.is_floating_point(): data = data.float()
        size_target = self._get_resize_target(size)
        align_corners = None if interp_mode=='nearest' else False
        data_new = torch.nn.functional.interpolate(data, size=size_target, mode=interp_mode, align_corners=align_corners)
        if convert_types: data_new = data_new.to(dtype=source_dtype)
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
            if tfm in size_tfms:
                crop_target = self._get_duration_crop_target(duration)
                x = tfm(x, size=crop_target, padding_mode=padding_mode)
            else:
                x = tfm(x)
        # below is the resizing part, `separate from cropping`
        if size is not None:
            # read target size from size dictionary, passed to transform method, default to own length (no resize)
            orig_size = x.shape[-1]
            new_size = size.get(getattr(self.__class__, '__name__'), orig_size)
            x.resize(new_size)
            # if x.config is not None: x.config._sr *= new_size/orig_size
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

    def _get_signal(self):
        if self.path is not None:
            self.sig, self._sr = torchaudio.load(self.path)
        elif self.spectro is not None and self.reconstruct_signal:
            # no signal or path but spectro is defined
            self.sig = self.config.mel2sig(self.spectro)

    def _load_spectro(self):
        if self.path is None: raise ValueError("item path wasn't provided")
        cache_path = self.config.get_cache_path(self.path)
        if cache_path.exists():
            spectro = torch.load(cache_path)
        else:
            # self.validate_consistencies(self.config)
            spectro = self._create_spectro()
            if self.config.cache:
                self.config.save_in_cache(self.path, spectro)
        self.spectro = spectro

    @property
    def loudness(self):
        if self._loudness is None:
            self._loudness = self._evaluate_loudness(self.sig, self.sr)
        return self._loudness

    @property
    def sig_raw(self):
        """Raw signal w/o preprocessing on optional loading"""
        if self._sig is None: self._get_signal()
        return self._sig

    @property
    def sig(self):
        """The default signal accessing method. Uses preprocessing."""
        if self._sig is None:
            self._get_signal()
        if self._sig is not None and not self.is_preprocessed: self._preprocess()
        return self._sig

    @sig.setter
    def sig(self, sig):
        self._sig = sig
        self._loudness = None
        # self._loudness = self._evaluate_loudness(self.sig, self.sr)

    @property
    def sr(self):
        if self._sr is None:
            # to load signal with preprocessing if needed
            self.sig
        return self._sr

    @sr.setter
    def sr(self, sr): self._sr = sr

    @property
    def spectro(self):
        if self._spectro is None: self._load_spectro()
        return self._spectro

    @spectro.setter
    def spectro(self, x): self._spectro = x

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
        return Audio(data=self.sig.squeeze().numpy(), rate=self.sr)

    @property
    def duration(self):
        if self._sig is not None:
            return self.n_samples / self.sr
        elif self.path is not None:
            sig_info, _ = torchaudio.info(self.path.as_posix())
            return sig_info.length/sig_info.rate
        elif self.spectro is not None:
            return (self.spectro.shape[-1] * self.config.sg_cfg.hop_length) / self.sr

    @property
    def n_samples(self):
        return self.sig_raw.shape[-1]

    @property
    def nchannels(self):
        if self.sig_raw is not None:
            return self.sig_raw.shape[-2]
        elif self.spectro.ndim == 3:
            return self.spectro.shape[-3]
        else:
            warnings.warn('Guessing channel number to be 1')
            return 1

