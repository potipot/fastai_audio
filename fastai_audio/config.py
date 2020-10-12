import hashlib
import os
import warnings
from dataclasses import dataclass, asdict, field
from pathlib import Path
import torch
import torchaudio
from fastai.core import ifnone
from fastai.data_block import get_files
from fastprogress.fastprogress import progress_bar
from torchaudio.transforms import Spectrogram, MelScale, MFCC, AmplitudeToDB


def md5(s):
    s = str(s)
    return hashlib.md5(s.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SpectrogramConfig:
    '''Configuration for how Spectrograms are generated'''
    f_min: float = 0.0
    f_max: float = None
    n_fft: int = 2560
    n_mels: int = 128
    pad: int = 0
    to_db_scale: bool = True
    top_db: int = 100
    n_mfcc: int = 20
    win_length: int = None
    hop_length: int = None
    n_stft: int = None

    def __post_init__(self):
        """Assign some assumed values if those are not provided. This hack allows
        the class to be called with frozen=True"""
        if self.win_length is None: object.__setattr__(self, 'win_length', self.n_fft)
        if self.hop_length is None: object.__setattr__(self, 'hop_length', self.win_length // 2)
        if self.n_stft is None: object.__setattr__(self, 'n_stft', self.n_fft // 2 + 1)
        # self.win_length = ifnone(self.win_length, self.n_fft)
        # self.hop_length = ifnone(self.hop_length, self.win_length // 2)
        # self.n_stft = ifnone(self.n_stft, self.n_fft // 2 + 1)

    @property
    def melkwargs(self):
        d = self.mel_args
        d.update(self.spec_args); d.pop('n_stft')
        return d

    @property
    def mfcc_args(self):
        return {name: getattr(self, name) for name in ['n_mfcc', 'melkwargs']}

    @property
    def mel_args(self):
        return {name: getattr(self, name) for name in ["f_min", "f_max", "n_stft", "n_mels"]}

    @property
    def spec_args(self):
        return {name: getattr(self, name) for name in ["hop_length", "n_fft", "pad", "win_length"]}

    @property
    def inverse_mel_args(self):
        return {name: getattr(self, name) for name in ["n_stft", "n_mels", "f_min", "f_max"]}

    @property
    def griffin_lim_args(self):
        return {name: getattr(self, name) for name in ["n_fft", "hop_length", "win_length"]}


class SpectrogramFuncs:
    def __init__(self, sr: int, sg_cfg: SpectrogramConfig):
        self.sg_cfg = sg_cfg
        self.spec = Spectrogram(**sg_cfg.spec_args)
        self.to_mel = MelScale(sample_rate=sr, **sg_cfg.mel_args)
        self.mfcc = MFCC(sample_rate=sr, **sg_cfg.mfcc_args)
        self.to_db = AmplitudeToDB(top_db=sg_cfg.top_db)


class ReconstructSignal:
    def __init__(self, sr: int, sg_cfg: SpectrogramConfig):
        self.sg_cfg = sg_cfg
        self.mel2sig = torch.nn.Sequential(
                torchaudio.transforms.InverseMelScale(sample_rate=sr, **sg_cfg.inverse_mel_args),
                torchaudio.transforms.GriffinLim(**sg_cfg.griffin_lim_args)
            )

    def __call__(self, *args, **kwargs):
        return self.mel2sig(*args, **kwargs)


@dataclass
class AudioConfig:
    '''Options for pre-processing fastai_audio signals'''
    cache: bool = True
    cache_dir = Path.home()/'.fastai/cache'
    # force_cache = False >>> DEPRECATED Use clear cache instead

    duration: int = None
    max_to_pad: float = None
    pad_mode: str = "zeros"
    remove_silence: str = None
    pre_emphasis_coeff: float = 0.97
    use_spectro: bool = True
    mfcc: bool = False

    delta: bool = False
    silence_padding: int = 200
    silence_threshold: int = 20
    segment_size: int = None
    resample_to: int = None
    target_loudness: float = -23.0
    standardize: bool = False
    downmix: bool = True

    _processed = False
    _sr = None
    _nchannels = None

    sg_cfg: SpectrogramConfig = SpectrogramConfig()
    _sg_funcs: SpectrogramFuncs = field(repr=False, compare=False, default=None)
    _mel2sig: ReconstructSignal = field(repr=False, compare=False, default=None)

    # def __setattr__(self, name, value):
    #     '''Override to warn user if they are mixing seconds and ms'''
    #     if name in 'duration max_to_pad segment_size'.split():
    #         if value is not None and value <= 30:
    #             warnings.warn(f"{name} should be in milliseconds, it looks like you might be trying to use seconds")
    #     self.__dict__[name] = value

    @property
    def sg_funcs(self):
        if self._sg_funcs is None:
            self._sg_funcs = SpectrogramFuncs(self._sr, self.sg_cfg)
        else:
            assert(self._sg_funcs.sg_cfg == self.sg_cfg)
        return self._sg_funcs

    @property
    def mel2sig(self):
        if self._mel2sig is None:
            self._mel2sig = ReconstructSignal(self._sr, self.sg_cfg)
        else:
            assert(self._mel2sig.sg_cfg == self.sg_cfg)
        return self._mel2sig

    def clear_cache(self):
        '''Delete the files and empty dirs in the cache folder'''
        num_removed = 0
        parent_dirs = set()
        if not os.path.exists(self.cache_dir/"cache_contents.txt"):
            print("Cache contents not found, try calling again after creating your AudioList")

        with open(self.cache_dir/"cache_contents.txt", 'r') as f:
            pb = progress_bar(f.read().split('\n')[:-1])
            for line in pb:
                if not os.path.exists(line): continue
                else:
                    try:
                        os.remove(line)
                    except Exception as e:
                        print(f"Warning: Failed to remove {line}, due to error {str(e)}...continuing")
                    else:
                        parent = Path(line).parents[0]
                        parent_dirs.add(parent)
                        num_removed += 1
        for parent in parent_dirs:
            if(os.path.exists(parent) and len(parent.ls()) == 0):
                try:
                    os.rmdir(str(parent))
                except Exception as e:
                    print(f"Warning: Unable to remove empty dir {parent}, due to error {str(e)}...continuing")
        os.remove(self.cache_dir/"cache_contents.txt")
        print(f"{num_removed} files removed")

    def cache_size(self):
        '''Check cache size, returns a tuple of int in bytes, and string representing MB'''
        cache_size = 0
        if not os.path.exists(self.cache_dir):
            print("Cache not found, try calling again after creating your AudioList")
            return (None, None)
        for (path, dirs, files) in os.walk(self.cache_dir):
            for file in files:
                cache_size += os.path.getsize(os.path.join(path, file))
        return (cache_size, f"{cache_size//(2**20)} MB")

    def record_cache_contents(self, files):
        '''Writes cache filenames to log for safe removal using 'clear_cache()' '''
        try:
            with open(self.cache_dir / "cache_contents.txt", 'a+') as f:
                for file in files:
                    f.write(str(file) + '\n')
        except Exception as e:
            print(f"Unable to save files to cache log, cache at {self.cache_dir} may need to be cleared manually")

    def get_cache_path(self, fn:Path):
        # folder = md5(str(asdict(self))+str(asdict(self.sg_cfg)))
        folder = md5(self.__repr__())
        fname = f"{md5(fn)}-{fn.name}.pt"
        return Path(self.cache_dir/(f"{folder}/{fname}"))

    def save_in_cache(self, fn, spectro):
        cache_path = self.get_cache_path(fn)
        os.makedirs(cache_path.parent, exist_ok=True)
        torch.save(spectro, cache_path)
        self.record_cache_contents([cache_path])


def get_cache(config, cache_type, item_path, params):
    if not config.cache_dir: return None
    details = "-".join(map(str, params))
    top_level = config.cache_dir / f"{cache_type}_{details}"
    subfolder = f"{item_path.name}-{md5(item_path)}"
    mark = top_level/subfolder
    files = get_files(mark) if mark.exists() else None
    return files


def make_cache(sigs, sr, config, cache_type, item_path, params):
    details = "-".join(map(str, params))
    top_level = config.cache_dir / f"{cache_type}_{details}"
    subfolder = f"{item_path.name}-{md5(item_path)}"
    mark = top_level/subfolder
    files = []
    if len(sigs) > 0:
        os.makedirs(mark, exist_ok=True)
        for i, s in enumerate(sigs):
            if s.shape[-1] < 1: continue
            fn = mark/(str(i) + '.wav')
            files.append(fn)
            torchaudio.save(str(fn), s, sr)
    return files