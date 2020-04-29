from annotations import sequences
from .audio import *
from .transform import *
from pathlib import Path as PosixPath
from typing import Tuple, Callable
from IPython.core.debugger import set_trace
from collections import Counter
from dataclasses import dataclass, asdict
import hashlib
import matplotlib as plt
from fastai.vision import *
from fastprogress.fastprogress import progress_bar
import torchaudio
import warnings
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, MFCC, Spectrogram, MelScale
import loudnorm

class EmptyFileException(Exception):
    pass


def md5(s):
    return hashlib.md5(s.encode("utf-8")).hexdigest()


class AudioDataBunch(DataBunch):
    def show_batch_stats(self, include_edges=False):
        lens = np.concatenate([sequences(batch, include_edges=include_edges, ignore_values=self.c2i['sil']) for _, y in self.train_dl for batch in y])
        fig, ax = plt.subplots(1)

        ax.hist(lens, bins=lens.max().astype('int'))
        ax.set_xlabel('Sequence length')
        ax.set_ylabel('No of counts')
        ax.set_xticks(np.arange(lens.max(), step=5))
        return lens


@dataclass
class SpectrogramConfig:
    '''Configuration for how Spectrograms are generated'''
    f_min: float = 0.0
    f_max: float = None
    n_fft: int = 2560
    win_length: int = None
    hop_length: int = None
    n_mels: int = 128
    pad: int = 0
    to_db_scale: bool = True
    top_db: int = 100
    n_mfcc: int = 20
    n_stft:int = None

    def __post__init__(self):
        """Assign some assumed values if those are not provided"""
        self.win_length = ifnone(self.win_length, self.n_fft)
        self.hop_length = ifnone(self.hop_length, self.win_length // 2)
        self.n_stft = ifnone(self.n_stft, self.n_fft // 2 + 1)

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


@dataclass
class AudioConfig:
    '''Options for pre-processing audio signals'''
    cache: bool = True
    cache_dir = Path.home()/'.fastai/cache'
    # force_cache = False >>> DEPRECATED Use clear cache instead
    
    duration: int = None
    max_to_pad: float = None
    pad_mode: str = "zeros"
    remove_silence: str = None
    use_spectro: bool = True
    mfcc: bool = False
    
    delta: bool = False
    silence_padding: int = 200
    silence_threshold: int = 20
    segment_size: int = None
    resample_to: int = None
    standardize: bool = False
    downmix: bool = False

    _processed = False
    _sr = None
    _nchannels = None
    
    sg_cfg: SpectrogramConfig = SpectrogramConfig()
        
    def __setattr__(self, name, value):
        '''Override to warn user if they are mixing seconds and ms'''
        if name in 'duration max_to_pad segment_size'.split():
            if value is not None and value <= 30:
                warnings.warn(f"{name} should be in milliseconds, it looks like you might be trying to use seconds")
        self.__dict__[name] = value

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
    
def get_cache(config, cache_type, item_path, params):
    if not config.cache_dir: return None
    details = "-".join(map(str, params))
    top_level = config.cache_dir / f"{cache_type}_{details}"
    subfolder = f"{item_path.name}-{md5(str(item_path))}"
    mark = top_level/subfolder
    files = get_files(mark) if mark.exists() else None
    return files

def make_cache(sigs, sr, config, cache_type, item_path, params):
    details = "-".join(map(str, params))
    top_level = config.cache_dir / f"{cache_type}_{details}"
    subfolder = f"{item_path.name}-{md5(str(item_path))}"
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

def downmix_item(item, config, path):
    item_path, label = item
    if isinstance(item_path, str): item_path = PosixPath(item_path)
    if not os.path.exists(item_path): item_path = path/item_path
    files = get_cache(config, "dm", item_path, [])
    if not files:
        sig, sr = torchaudio.load(item_path)
        sig = [tfm_downmix(sig)]
        files = make_cache(sig, sr, config, "dm", item_path, [])
        _record_cache_contents(config, files)
    return list(zip(files, [label]*len(files)))

def resample_item(item, config, path):
    item_path, label = item
    if isinstance(item_path, str): item_path = PosixPath(item_path)
    if not os.path.exists(item_path): item_path = path/item_path
    sr_new = config.resample_to
    files = get_cache(config, "rs", item_path, [sr_new])
    if not files:
        sig, sr = torchaudio.load(item_path)
        sig = [tfm_resample(sig, sr, sr_new)]
        files = make_cache(sig, sr_new, config, "rs", item_path, [sr_new])
        _record_cache_contents(config, files)
    return list(zip(files, [label]*len(files)))

def remove_silence(item, config, path):
    item_path, label = item
    if isinstance(item_path, str): item_path = PosixPath(item_path)
    if not os.path.exists(item_path): item_path = path/item_path
    st, sp = config.silence_threshold, config.silence_padding
    remove_type = config.remove_silence
    cache_prefix = f"sh-{remove_type[0]}"
    files = get_cache(config, cache_prefix, item_path, [st, sp])
    if not files:
        sig, sr = torchaudio.load(item_path)
        sigs = tfm_remove_silence(sig, sr, remove_type, st, sp)
        files = make_cache(sigs, sr, config, cache_prefix, item_path, [st, sp])
        _record_cache_contents(config, files)
    return list(zip(files, [label]*len(files)))

def segment_items(item, config, path):
    item_path, label = item
    if isinstance(item_path, str): item_path = PosixPath(item_path)
    if not os.path.exists(item_path): item_path = path/item_path
    files = get_cache(config, "s", item_path, [config.segment_size])
    if not files:
        sig, sr = torchaudio.load(item_path)
        segsize = int(config._sr*config.segment_size/1000)
        sigs = []
        siglen = sig.shape[-1]
        for i in range((siglen//segsize) + 1):
            #if there is a full segment, add it, if not take the remaining part and zero pad to correct length
            if((i+1)*segsize <= siglen): sigs.append(sig[:,i*segsize:(i+1)*segsize])
            else: sigs.append(torch.cat([sig[:,i*segsize:], torch.zeros(sig.shape[0],segsize-sig[:,i*segsize:].shape[-1])],dim=1))
        files = make_cache(sigs, sr, config, "s", item_path, [config.segment_size])
        _record_cache_contents(config, files)
    return list(zip(files, [label]*len(files)))

def _record_cache_contents(cfg, files):
    '''Writes cache filenames to log for safe removal using 'clear_cache()' '''
    try:
        with open(cfg.cache_dir/"cache_contents.txt", 'a+') as f:
            for file in files: 
                f.write(str(file)+'\n')
    except Exception as e:
        print(f"Unable to save files to cache log, cache at {cfg.cache_dir} may need to be cleared manually")

def get_outliers(len_dict, devs):
    np_lens = array(list(len_dict.values()))
    stdev = np_lens.std()
    lower_thresh = np_lens.mean() - stdev*devs
    upper_thresh = np_lens.mean() + stdev*devs
    outliers = [(k,v) for k,v in len_dict.items() if not (lower_thresh < v < upper_thresh)]
    return sorted(outliers, key=lambda tup: tup[1])

def _set_sr(item_path, config, path):
    # a bit hacky, this is to make audio_predict work when an AudioItem arrives instead of path
    if isinstance(item_path, AudioItem): item_path = item_path.path
    if not os.path.exists(item_path): item_path = path/item_path
    sig, sr = torchaudio.load(item_path)
    config._sr = sr

def _set_nchannels(item_path, config, path):
    # Possibly should combine with previous def, but wanted to think more first
    if isinstance(item_path, AudioItem):
        raise TypeError
        item_path = item_path.path
    if not os.path.exists(item_path): item_path = path/item_path
    info = loudnorm.get_file_info(item_path)
    config._nchannels = info.channels

class AudioLabelList(LabelList):

    def _pre_process(self):
        x, y = self.x, self.y
        cfg = x.config
        
        if len(x.items) > 0:
            if not cfg.resample_to: _set_sr(x.items[0], x.config, x.path)
            if cfg._nchannels is None: _set_nchannels(x.items[0], x.config, x.path)
            if cfg.downmix or cfg.remove_silence or cfg.segment_size or cfg.resample_to:
                items = list(zip(x.items, y.items))

                def concat(x, y): return np.concatenate(
                    (x, y)) if len(y) > 0 else x
                
                if x.config.downmix:
                    print("Preprocessing: Downmixing to Mono")
                    cfg._nchannels=1
                    items = [downmix_item(i, x.config, x.path) for i in progress_bar(items)]
                    items = reduce(concat, items, np.empty((0, 2)))

                if x.config.resample_to:
                    print("Preprocessing: Resampling to", x.config.resample_to)
                    cfg._sr = x.config.resample_to 
                    items = [resample_item(i, x.config, x.path) for i in progress_bar(items)]
                    items = reduce(concat, items, np.empty((0, 2)))

                if x.config.remove_silence:
                    print("Preprocessing: Removing Silence")
                    items = [remove_silence(i, x.config, x.path) for i in progress_bar(items)]
                    items = reduce(concat, items, np.empty((0, 2)))

                if x.config.segment_size:
                    print("Preprocessing: Segmenting Items")
                    items = [segment_items(i, x.config, x.path) for i in progress_bar(items)]
                    items = reduce(concat, items, np.empty((0, 2)))

                nx, ny = tuple(zip(*items))
                x.items, y.items = np.array(nx), np.array(ny)
 
        self.x, self.y = x, y
        self.y.x = x
   
    def process(self, *args, **kwargs):
        self._pre_process()
        # TODO this is a hack to always use tfm_crop_time
        #  consider removing the _processed attribute and move crop_time
        #  to proper apply_tfms place
        #  also accessed in fastai_audio.audio.data@line357
        self.x.config._processed = True
        super().process(*args, **kwargs)


class AudioList(ItemList):
    _label_list = AudioLabelList
    _bunch = AudioDataBunch
    config: AudioConfig

    def _filter_empty(items):
        def _filter(fn: Path) -> bool:
            return os.path.exists(fn) and os.path.getsize(fn) > 0

        old_count = len(items)
        items = list(filter(_filter, items))
        new_count = len(items)
        if old_count != new_count:
            print(f"Filtered out {old_count-new_count} empty files")
        return items

    def __init__(self, items, path, config=AudioConfig(), **kwargs):
        items = AudioList._filter_empty(items)
        super().__init__(items, path, **kwargs)
        cd = config.cache_dir
        # After calling init from super class the _label_list fields gets overwritten, thats why its re-initialized
        self._label_list = self.__class__._label_list
        # wants to store Audio label list here to shadow the .process method and use audio + config preprocessing?
        # why not use preprocessor for that?
        if str(path) not in str(cd): 
            config.cache_dir = path / cd
        self.config = config
        self.copy_new += ['config']
        self._sr = self.register_sampling_rate()
        self._funcs = self._create_spectro_funcs()

    def open(self, fn): # file name, it seems
        fn = Path(fn)
        if self.path is not None and not fn.exists() and str(self.path) not in str(fn): fn = self.path/item
        if self.config.use_spectro:
            item = self.add_spectro(fn)
        else:
            raise NotImplementedError
            func_to_add = self._get_pad_func() if self.config.max_to_pad or self.config.segment_size else None
            item = AudioItem.create(fn, func_to_add)
            self._validate_consistencies(item)
        return item

    def _validate_consistencies(self, item):
        if(self.config._sr is not None and item.sr != self.config._sr):
            raise ValueError(f'''Multiple sample rates detected. Sample rate {item.sr} of file {item.path} 
                                does not match config sample rate {self.config._sr} 
                                this means your dataset has multiple different sample rates, 
                                please choose one and set resample_to to that value''')
        if(self.config._nchannels is not None and self.config._nchannels != item.nchannels):
            raise ValueError(f'''Multiple channel sizes detected. Channel size {item.nchannels} of file 
                                {item.path} does not match others' channel size of {self.config._nchannels}. A dataset may
                                not contain different number of channels. Please set downmix=true in AudioConfig or 
                                separate files with different number of channels.''')

    def add_spectro(self, fn:PathOrStr):
        cache_path = self._get_cache_path(fn)
        func_to_add = self._get_pad_func() if self.config.max_to_pad or self.config.segment_size else None
        """Here audio file is loaded immediately after creation"""
        item = AudioItem.create(fn, func_to_add)
        self._validate_consistencies(item)
        if self.config.cache and cache_path.exists():
            spectro = torch.load(cache_path)
        else:
            #Dropping sig and sr off here, should I propogate this to new audio item if I have it?
            spectro = self.create_spectro(item)
            if self.config.cache:
                self._save_in_cache(fn, spectro)
        item.spectro = spectro
        return item

    def _get_pad_func(self):
        def pad_func(sig, sr): 
            pad_len = self.config.max_to_pad if self.config.max_to_pad is not None else self.config.segment_size
            num_samples = int((sr*pad_len)/1000)
            return tfm_padtrim_signal(sig, num_samples, pad_mode="zeros")
        return pad_func

    def _create_spectro_funcs(self):
        cfg = self.config.sg_cfg
        return {
            'spec': Spectrogram(**cfg.spec_args),
            'to_mel': MelScale(sample_rate=self._sr, **cfg.mel_args),
            'mfcc': MFCC(sample_rate=self._sr, **cfg.mfcc_args),
            'to_db': AmplitudeToDB(top_db=cfg.top_db)
        }

    def create_spectro(self, item:AudioItem):
        assert(item.sr == self._sr), f'AudioItem has different sr({item.sr}) than the ItemList{self._sr}, forgot to resample?'
        if self.config.mfcc:
            spec = self._funcs['mfcc'](item.sig)
        else:
            spec = self._funcs['spec'](item.sig)
            spec = self._funcs['to_mel'](spec)
            if self.config.sg_cfg.to_db_scale: 
                spec = self._funcs['to_db'](spec)
        spec = spec.detach()
        if self.config.standardize: 
            spec = standardize(spec)
        if self.config.delta: 
            spec = torch.cat([torch.stack([m,torchdelta(m),torchdelta(m, order=2)]) for m in spec])
        return spec

    def _get_cache_path(self, fn:Path):
        folder = md5(str(asdict(self.config))+str(asdict(self.config.sg_cfg)))
        fname = f"{md5(str(fn))}-{fn.name}.pt"
        return Path(self.config.cache_dir/(f"{folder}/{fname}"))

    def _save_in_cache(self, fn, spectro):
        cache_path = self._get_cache_path(fn)
        os.makedirs(cache_path.parent, exist_ok=True)
        torch.save(spectro, cache_path)
        _record_cache_contents(self.config, [cache_path])

    def get(self, i):
        fn = super().get(i)
        return self.open(fn)

    def reconstruct(self, x): return x

    def stats(self, prec=0, devs=3, figsize=(15,5)):
        '''Displays samples, plots file lengths and returns outliers of the AudioList'''
        len_dict = {}
        rate_dict = {}
        pb = progress_bar(self)
        for item in pb:
            si, ei = torchaudio.info(str(item.path))
            len_dict[item.path] = si.length/si.rate
            rate_dict[item.path] = si.rate
        lens = list(len_dict.values())
        rates = list(rate_dict.values())
        print("Sample Rates: ")
        for sr,count in Counter(rates).items(): print(f"{int(sr)}: {count} files")
        self._plot_lengths(lens, prec, figsize)
        return len_dict
    
    def _plot_lengths(self, lens, prec, figsize):
        '''Plots a list of file lengths displaying prec digits of precision'''
        rounded = [round(i, prec) for i in lens]
        rounded_count = Counter(rounded)
        plt.figure(num=None, figsize=figsize, dpi=80, facecolor='w', edgecolor='k')
        labels = sorted(rounded_count.keys())
        values = [rounded_count[i] for i in labels]
        width = 1
        plt.bar(labels, values, width)
        xticks = np.linspace(int(min(rounded)), int(max(rounded))+1, 10)
        plt.xticks(xticks)
        plt.show()
  
    @classmethod
    def from_folder(cls, path:PathOrStr='.', extensions:Collection[str]=None, **kwargs)->ItemList:
        "Get the list of files in `path` that have an audio suffix. `recurse` determines if we search subfolders."
        extensions = ifnone(extensions, AUDIO_EXTENSIONS)
        return super().from_folder(path=path, extensions=extensions, **kwargs)
        
        
    @classmethod
    def from_df(cls, df:DataFrame, path:PathOrStr, cols:IntsOrStrs=0, folder:PathOrStr=None, suffix:str='', **kwargs)->ItemList:
        "Get the filenames in `cols` of `df` with `folder` in front of them, `suffix` at the end."
        suffix = suffix or ''
        res = super().from_df(df, path=path, cols=cols, **kwargs)


        pref = f'{res.path}{os.path.sep}'
        if folder is not None: pref += f'{folder}{os.path.sep}'
        res.items = np.char.add(np.char.add(pref, res.items.astype(str)), suffix)
        return res

    def register_sampling_rate(self):
        if self.config.resample_to is None:
            _, sr = torchaudio.load(np.random.choice(self.items))
        else:
            sr = self.config.resample_to
        return sr


def open_audio(fn: Path, after_open: Callable = None) -> AudioItem:
    try:
        sig, sr = torchaudio.load(fn)
        if after_open:
            sig = after_open(sig, sr)
        return AudioItem(sig=sig, sr=sr, path=fn)
    except RuntimeError as e:
        err_msg = f"Error loading file at {fn}"
        if os.path.exists(fn) and os.path.getsize(fn) == 0:
            err_msg = f"{err_msg} of size 0 with Torch Audio: {e}"
            raise EmptyFileException(err_msg)
        else:
            err_msg = f"{err_msg} with Torch Audio: {e}"
            raise RuntimeError(err_msg)

