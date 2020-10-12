import io
import json
import subprocess
from pathlib import Path

import pyloudnorm
import torch
import numpy as np

from fastai_audio.fastai_audio.utils import _channel_first, _to_numpy, get_file_info

_num_backends = {np.ndarray: np, torch.Tensor: torch}


def _clip_hard(signal):
    if isinstance(signal, torch.Tensor):
        return signal.clamp(-1, 1)
    elif isinstance(signal, np.ndarray):
        return signal.clip(-1, 1)
    else:
        raise TypeError


def _clip_none(signal):
    return signal


def _clip_soft(signal, alpha=1.2):
    return torch.tanh(alpha * signal)


def _clip_soft_adaptive(signal, h=0.85):
    # get numerical backend to process signal
    nb = _num_backends.get(type(signal))
    alpha = 0.5 / h * np.log((1 + h) / (1 - h))
    func = nb.tanh(alpha * signal)
    return nb.where(nb.abs(func) < nb.abs(signal), func, signal)


def clip(clipping_method: str, signal, **kwargs):
    clipping_methods = {'hard': _clip_hard,
                        'soft': _clip_soft,
                        'soft_adaptive': _clip_soft_adaptive,
                        None: _clip_none}
    _clip = clipping_methods.get(clipping_method, _clip_none)
    return _clip(signal, **kwargs)


def _run_ffmpeg_process(filepath:Path, target:float = -23.0):
    if filepath.suffix != '.wav': raise NotImplementedError
    info = get_file_info(filepath)
    sr, n = info.rate, info.length
    ffargs = ['ffmpeg', '-loglevel', 'level',
              '-i', filepath.as_posix(),
              '-hide_banner',
              '-filter',
              f'loudnorm=I={target}:dual_mono=true:print_format=json,aresample=ocl=mono:osr={sr}',
              '-f',
              'wav',
              'pipe:1']

    pipe = subprocess.run(ffargs, capture_output=True)
    stream = io.BytesIO(pipe.stdout)
    message = json.loads(pipe.stderr.decode().split('[info]')[-1])
    return stream, message


def volume(sig, input_loudness, target_loudness):
    # calculate the gain needed to scale to the desired loudness level
    delta_loudness = target_loudness - input_loudness
    gain = 10.0 ** (delta_loudness / 20.0)
    return gain * sig


def get_loudness(signal, sr):
    meter = pyloudnorm.Meter(sr)
    sig = _to_numpy(signal)
    sig = _channel_first(sig)
    loudness = meter.integrated_loudness(sig)
    return loudness
