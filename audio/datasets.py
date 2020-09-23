import pickle
import json
import re
import pandas as pd
from pathlib import Path
from fastprogress import progress_bar


class Dataset:
    def __init__(self, directory: Path, sample=None, force_reload=False):
        self.directory = directory
        self.sample = sample
        self.force_reload = force_reload
        if hasattr(self, 'name'): self.__dict_file = self.directory / (self.name + '_dict.pkl')
        self.__labels_dict = None

    def reload(self):
        self.__labels_dict = self.read_labels()
        self.save_dict(self.__labels_dict, self.__dict_file)

    def read_labels(self) -> dict:
        raise NotImplementedError("You have to define dictionary generation method if you want to inherit this class")

    @staticmethod
    def load_dict(file_name: Path) -> dict:
        assert (file_name.suffix == '.pkl')
        with open(file_name.as_posix(), 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def save_dict(dict_obj: dict, file_name: Path):
        assert (file_name.suffix == '.pkl')
        with open(file_name.as_posix(), 'wb') as f:
            pickle.dump(dict_obj, f, pickle.HIGHEST_PROTOCOL)

    @property
    def labels(self) -> dict:
        if self.force_reload or not self.__dict_file.exists():
            self.reload()
            self.force_reload = False
        else:
            self.__labels_dict = self.load_dict(self.__dict_file)
        return self.__labels_dict


class MozillaDataset(Dataset):
    name = 'mozilla'

    def read_labels(self) -> dict:
        labels_df = pd.read_csv(self.directory / "validated.tsv", sep='\t')[['path', 'sentence']]
        labels_df['path'] = labels_df['path'].apply(lambda filepath: (self.directory / 'clips' / filepath).as_posix())
        return labels_df.set_index('path')['sentence'].to_dict()


class ToucanTestDataset(Dataset):
    name = 'toucan_test'

    def __init__(self, *args, no_numerics=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels_file = self.directory / ('labels_no_numerics.txt' if no_numerics else 'labels.txt')

    def read_labels(self) -> dict:
        with open(self.labels_file.as_posix(), encoding='utf-8') as file:
            lines = file.read().splitlines()
        # create dict with line numbers as keys
        numerical_dict = {i: line for i, line in enumerate(lines, start=1)}

        # match pattern for line with 1-3 digits
        pattern = re.compile('line_(\d{1,3})_')
        filename_dict = dict()
        for audio_file in self.directory.glob('*/*.wav'):
            audio_path = audio_file.as_posix()
            # extract line number from file name
            line_number = int(pattern.findall(audio_path)[0])
            filename_dict[audio_path] = numerical_dict[line_number]
        return filename_dict


class IplaDataset(Dataset):
    name = 'ipla'
    intro_lines = ['Z niepokoju drżę więc obejmij mnie dobry Wrocławiu',
                   'Z marzeń utwórz mgłę bym schroniła się bym ukryła',
                   'Co przyniesie czas zauroczy nas mądry Wrocławiu',
                   'Stoję u twych drzwi Czy otworzy mi pierwsza miłość',
                   'Jestem tutaj przytul mnie',
                   'Pierwsze tak pierwsze nie',
                   'Będę obok po to by',
                   'Spełnić twoje sny piękne sny Dobre sny',
                   'Ooooo Stoję u twych drzwi',
                   'Czy otworzy mi',
                   'Pierwsza miłość']

    def read_labels(self):
        labels_df = pd.read_csv(self.directory / "klipy_normalized.csv", sep=';', encoding='windows-1250', names=['path', 'sentence'])
        # skip intro lines
        labels_df = labels_df[~labels_df['sentence'].str.lower().isin([line.lower() for line in self.intro_lines])]
        # get absolute path
        labels_df['path'] = labels_df['path'].apply(lambda filepath: (self.directory / 'video_clips' / filepath)
                                                    .with_suffix('.wav')
                                                    .as_posix())
        return labels_df.set_index('path')['sentence'].to_dict()


class ClarinDataset(Dataset):
    name = 'clarin'

    def extract_string(self, json_file) -> str:
        with open(json_file.as_posix(), encoding='utf-8') as json_cached:
            data = json.load(json_cached)

        annotations_raw = list()
        # read words dict
        for level in data['levels']:
            if level['name'] == 'Word':
                annotations_raw = level['items']

        # extract word value from dict
        word_list = [word['labels'][0]['value'] for word in annotations_raw]

        # join to sentence and return
        return " ".join(word_list)

    def read_labels(self) -> dict:
        clarin_dict = {}
        wavfiles = list(self.directory.glob('**/*.wav'))
        pb = progress_bar(wavfiles)
        pb.comment = 'parsing jsons'
        for wav_file in pb:
            wav_name = wav_file.stem
            json_file = wav_file.with_name(wav_name + '_annot').with_suffix(".json")
            if json_file.exists():
                annotation = self.extract_string(json_file)
                clarin_dict[wav_file.as_posix()] = annotation
        return clarin_dict


class LunaDataset(Dataset):
    name = 'luna'

    def read_labels(self) -> dict:
        read = pd.read_csv(self.directory/'labels.csv', names=['filepath', 'label'])
        read['filepath'] = read['filepath'].apply(lambda path: (self.directory / path).as_posix())
        return read.set_index('filepath')['label'].to_dict()

