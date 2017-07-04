import sugartensor as tf

from data.base_data_loader import BaseDataLoader
from data.preprocessors.conll_preprocessor import ConllPreprocessor

__author__ = 'georgi.val.stoyan0v@gmail.com'


class ConllLoader(BaseDataLoader):
    _name = 'ConllLoader'
    TSV_DELIM = '\t'
    DATA_COLUMN = ConllPreprocessor.EXAMPLE_COLUMN

    DEFAULT_META_DATA_FILE = 'metadata_labeledTrainData.tsv'
    DEFAULT_METADATA_DIR = 'data/datasets/conll_2003/'

    def __init__(self, bucket_boundaries, file_names, *args, **kwargs):
        self.__file_preprocessor = None

        self.field_delim = ConllLoader.TSV_DELIM
        self.file_names = file_names

        record_defaults = [[''], [''], [''], [''], ['']]
        skip_header_lines = 1
        data_column = ConllLoader.DATA_COLUMN

        super().__init__(record_defaults, self.field_delim, data_column, bucket_boundaries, file_names, *args,
                         skip_header_lines=skip_header_lines, meta_file=ConllLoader.DEFAULT_META_DATA_FILE,
                         save_dir=ConllLoader.DEFAULT_METADATA_DIR, **kwargs)

        self.source_words, self.source_pos, self.source_chunk, self.source_capitals, self.target,\
            self.o1, self.o2, self.o3, self.o4, self.o5= self.get_data()

        data_size = self.data_size
        self.num_batches = data_size // self._batch_size
