from collections import Counter
from functools import reduce
from pathlib import Path

import numpy as np
import torch
import torchaudio
from fastai.core import listify, warnings


def _channel_first(sig):
    """Use the hack that maximum channel number is 5"""
    return sig.T if sig.shape[1] > 5 else sig


def _signal_first(sig):
    """Use the hack that maximum channel number is 5"""
    return sig.T if sig.shape[0] > 5 else sig


def _to_numpy(array):
    if isinstance(array, torch.Tensor):
        res = array.numpy()
    elif isinstance(array, np.ndarray):
        res = array
    else:
        raise TypeError(f'conversion unsupported for {type(array)}')
    return res


def one_hot_ndarray(signal, n_classes):
    assert(signal.ndim == 1)
    res = np.zeros(shape=(n_classes, signal.size), dtype=np.float32)
    res[signal, np.arange(signal.size)] = 1.0
    return res


def one_hot_decode(tens:torch.Tensor, axis=-2):
    maxes, indices = torch.max(tens, dim=axis)
    # below tests if there is no equal items along chosen axis. Number of zeros in an array should equal number of maxes
    if len((tens - maxes.unsqueeze(axis)).nonzero()) != tens.numel()-maxes.numel():
        warnings.warn('Tensor is not correctly one hot encoded, some max values repeat along chosen axis.')
        # TODO: use softmax, add blank symbol
    return indices


def one_hot_tensor(signal, n_classes):
    res = torch.zeros(size=(n_classes, signal.numel()), dtype=torch.float)
    res.scatter_(0, signal.view(1,-1).long(), 1.0)
    return res


def sequences(tens, include_edges=False, c2i: dict = None, return_counter=False, ignore_values=None):
    """Calculate the number of consecutive values in a tensor signal.
    If c2i mapping is provided, silence signal is ignored in stats.
    :param return_counter: """
    tens = tens.squeeze()
    if tens.ndim == 2:
        tens = one_hot_decode(tens)
    elif tens.ndim == 1:
        pass
    else:
        raise NotImplementedError('Unknown tensor shape')
    if include_edges:
        i, f = None, None
    else:
        i, f = 1, -1
    if ignore_values is not None:
        ignore_values = listify(ignore_values)
        if tens[0] in ignore_values: i = None
        if tens[-1] == ignore_values: f = None
        filter_tensors = (tens != val for val in ignore_values)
        filter = reduce(torch.mul, filter_tensors)
        tens = tens[filter]

    # TODO consider moving this to reconstruct
    s = ((tens[1:] - tens[:-1]).nonzero().squeeze() + 1).view(-1)
    split = s - torch.nn.functional.pad(s[:-1], (1, 0))
    chunks = torch.split(tens, [*split.tolist(), len(tens) - split.sum()])
    if return_counter:
        counter = [list(*Counter(chunk.tolist()).items()) for chunk in chunks[i:f]]
        return counter
    lengths = np.array([t.numel() for t in chunks[i:f]])
    # need to filter out zeros, which result from padding of empty tensor
    return lengths[lengths > 0]


def get_json(path:Path):
    path = Path(path)
    return path.parent / (path.stem + '_annot.json')


def get_file_info(p):
    if isinstance(p, Path): p = p.as_posix()
    signal_info, _ = torchaudio.info(p)
    return signal_info