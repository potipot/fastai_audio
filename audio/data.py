from . import AudioItem
from .config import get_cache, make_cache
from .transform import *
from pathlib import Path as PosixPath
import matplotlib as plt
from fastai.vision import *
from fastprogress.fastprogress import progress_bar
import torchaudio
from .utils import sequences, get_file_info
from .datasets import Dataset


class EmptyFileException(Exception):
    pass


class AudioDataBunch(DataBunch):
    def show_stats(self, ds_type: DatasetType = DatasetType.Valid, figsize=(10,10), log=True, bins=50, return_values=False):
        '''Displays samples, plots file lengths and returns outliers of the AudioList'''
        dl = self.dl(ds_type)
        pb = progress_bar(zip(dl.x, dl.y), len(dl.x))
        pb.comment = f'Analyzing {ds_type}'
        durations = []
        tempos = []
        rates = Counter()
        for x, y in pb:
            durations.append(x.duration)
            rates.update([x.sr])
            tempos.append(y.sig.shape[-1] / x.spectro.shape[-1])
        durations = np.array(durations)
        tempos = np.array(tempos)

        print("Sample Rates: ")
        for sr, count in rates.items(): print(f"{int(sr)}: {count} files")
        fig, axs = plt.subplots(2, 1, figsize=figsize)
        max_tempo = 0.25
        min_duration = 1.0
        max_duration = 15.0
        axs[0].hist(tempos, bins=bins, log=log)
        axs[0].axvline(max_tempo, color='r', alpha=0.5, label='max tempo')
        axs[0].plot([], [], ' ', label=f'{len(tempos[tempos < max_tempo])} samples in range')
        axs[0].plot([], [], ' ', label=f'{len(tempos[tempos >= max_tempo])} samples discarded')

        axs[0].set_title('tempos')
        axs[0].set_xlabel('char per frame')
        axs[0].legend(loc='upper right')

        axs[1].hist(durations, bins=bins, log=log)
        axs[1].axvline(min_duration, color='y', alpha=1, label='min duration')
        axs[1].axvline(max_duration, color='r', alpha=0.5, label='max duration')
        n_samples_in_range = len(durations[(durations < max_duration) * (durations > min_duration)])
        axs[1].plot([], [], ' ', label=f'{n_samples_in_range} samples in range')
        axs[1].plot([], [], ' ', label=f'{len(durations) - n_samples_in_range} samples discarded')

        axs[1].set_title('durations')
        axs[1].set_xlabel('time [s]')
        axs[1].legend(loc='upper right')
        plt.tight_layout()
        return tempos, durations if return_values else None

    def show_batch_stats(self, include_edges=False):
        lens = np.concatenate([sequences(batch, include_edges=include_edges, ignore_values=self.c2i['sil']) for _, y in self.train_dl for batch in y])
        fig, ax = plt.subplots(1)

        ax.hist(lens, bins=lens.max().astype('int'))
        ax.set_xlabel('Sequence length')
        ax.set_ylabel('No of counts')
        ax.set_xticks(np.arange(lens.max(), step=5))
        return lens


def downmix_item(item, config, path):
    item_path, label = item
    if isinstance(item_path, str): item_path = PosixPath(item_path)
    if not os.path.exists(item_path): item_path = path/item_path
    files = get_cache(config, "dm", item_path, [])
    if not files:
        sig, sr = torchaudio.load(item_path)
        sig = [tfm_downmix(sig)]
        files = make_cache(sig, sr, config, "dm", item_path, [])
        config.record_cache_contents(files)
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
        config.record_cache_contents(files)
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
        config.record_cache_contents(files)
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
        config.record_cache_contents(files)
    return list(zip(files, [label]*len(files)))

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
    info = get_file_info(item_path)
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

    @staticmethod
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

    def _get_pad_func(self):
        def pad_func(sig, sr): 
            pad_len = self.config.max_to_pad if self.config.max_to_pad is not None else self.config.segment_size
            num_samples = int((sr*pad_len)/1000)
            return tfm_padtrim_signal(sig, num_samples, pad_mode="zeros")
        return pad_func

    def get(self, i):
        file_name = super().get(i)
        return AudioItem(path=file_name, config=self.config)

    def reconstruct(self, x, **kwargs): return x
    
    def _plot_lengths(self, lens, prec, figsize, log=True, bins=10):
        '''Plots a list of file lengths displaying prec digits of precision'''
        rounded = [round(i, prec) for i in lens]
        rounded_count = Counter(rounded)
        plt.figure(num=None, figsize=figsize, dpi=80, facecolor='w', edgecolor='k')
        labels = sorted(rounded_count.keys())
        values = [rounded_count[i] for i in labels]
        width = 1
        plt.bar(labels, values, width, log=log, bins=bins)
        xticks = np.linspace(int(min(rounded)), int(max(rounded))+1, 10)
        plt.xticks(xticks)
        plt.show()
  
    @classmethod
    def from_folder(cls, path:Union[Dataset,Path,str]='.', extensions:Collection[str]=None, include=None, **kwargs)->ItemList:
        "Get the list of files in `path` that have an audio suffix. `recurse` determines if we search subfolders."
        # wrap to allow use of Dataset class explicitly
        if isinstance(path, Dataset):
            if include is None: include = path.sample
            path = path.directory
        extensions = ifnone(extensions, AUDIO_EXTENSIONS)
        return super().from_folder(path=path, extensions=extensions, include=include, **kwargs)
        
        
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
        self.config._sr = sr
        return sr


def open_audio(fn: Path, after_open: Callable = None) -> AudioItem:
    sig, sr = torchaudio.load(fn)
    if after_open:
        sig = after_open(sig, sr)
    return AudioItem(sig=sig, sr=sr, path=fn)


