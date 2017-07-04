import collections
import mmap

import pandas as pd
import re
from abc import abstractclassmethod


class BasePreprocessor(object):
    CLEAN_PREFIX = 'clean_'
    TEST_PREFIX = 'clean_test_'
    VOCABULARY_PREFIX = 'vocabulary_'
    VOCABULARY_POS = 'pos.tsv'
    VOCABULARY_CHUNK = 'chunk.tsv'
    VOCABULARY_ENTITY = 'entity.tsv'
    UNK_TOKEN_ID = 1

    _METADATA_PREFIX = 'metadata_'
    _PAD_TOKEN = '<PAD>'
    _UNK_TOKEN = '<UNK>'
    _EOS_TOKEN = '<EOS>'

    def __init__(self, path, filename, separator, vocabulary_size, max_data_length, pad_token=_PAD_TOKEN,
                 unk_token=_UNK_TOKEN, eos_token=_EOS_TOKEN):
        self._regex = re.compile('[%s]' % re.escape(r"""#"$%&'()*+/:;<=>@[\]^_`{|}~"""))
        self._remove_space_after_quote = re.compile(r'\b\'\s+\b')
        self._add_space = re.compile('([.,!?()-])')
        self._remove_spaces = re.compile('\s{2,}')
        self._dictionary = {}

        self.path = path
        self.filename = filename
        self.separator = separator
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.eos_token = eos_token
        self.vocabulary_size = vocabulary_size
        self.max_data_length = max_data_length
        self.test_split = 0
        self.data = None
        self.new_data = None
        self.data_size = 0
        self.test_size = 0

    def _build_dictionary(self, data, column_name, entity_column, pos_column=None, chunk_column=None):
        all_text = []

        for review in data[column_name]:
            all_text.extend(review.split())

        all_words = [(self.pad_token, -1), (self.unk_token, -1), (self.eos_token, -1)]

        assert all_words[BasePreprocessor.UNK_TOKEN_ID][0] == \
               self.unk_token, '<UNK> token id and actual position should match'

        all_words.extend(collections.Counter(all_text).most_common(self.vocabulary_size - 3))

        for word in all_words:
            if word[0] not in self._dictionary:
                self._dictionary[word[0]] = len(self._dictionary)

        word_column = 'Word'
        frequency_column = 'Frequency'
        metadata = pd.DataFrame(data=all_words, columns=[word_column, frequency_column])
        self.vocabulary_size = len(self._dictionary)

        print('Built vocabulary with size: %d' % self.vocabulary_size)

        metadata.to_csv(self.path + self._METADATA_PREFIX + self.filename, sep=self.separator, index=False,
                        encoding='utf-8')
        print('Saved vocabulary to metadata file')
        metadata[word_column].to_csv(self.path + self.VOCABULARY_PREFIX + self.filename, sep=self.separator,
                                     index=False, encoding='utf-8')
        print('Saved vocabulary to vocabulary file')

        entity_common = pd.DataFrame(data[entity_column].str.split().tolist()).stack().value_counts()
        entity_common = entity_common.to_frame().reset_index()
        entity_common = entity_common.rename(columns={'index': word_column, 0: frequency_column})

        entity_common[word_column].to_csv(self.path + self.VOCABULARY_PREFIX + self.VOCABULARY_ENTITY,
                                          sep=self.separator,
                                          index=False, encoding='utf-8')
        print('Saved entity vocabulary to vocabulary file')

        if pos_column is not None:
            pos_common = pd.DataFrame(data[pos_column].str.split().tolist()).stack().value_counts()
            pos_common = pos_common.to_frame().reset_index()
            pos_common = pos_common.rename(columns={'index': word_column, 0: frequency_column})

            pos_common[word_column].to_csv(self.path + self.VOCABULARY_PREFIX + self.VOCABULARY_POS, sep=self.separator,
                                           index=False, encoding='utf-8')

            print('Saved pos vocabulary to vocabulary file')

        if chunk_column is not None:
            chunk_common = pd.DataFrame(data[chunk_column].str.split().tolist()).stack().value_counts()
            chunk_common = chunk_common.to_frame().reset_index()
            chunk_common = chunk_common.rename(columns={'index': word_column, 0: frequency_column})

            chunk_common[word_column].to_csv(self.path + self.VOCABULARY_PREFIX + self.VOCABULARY_CHUNK,
                                             sep=self.separator,
                                             index=False, encoding='utf-8')
            print('Saved chunk vocabulary to vocabulary file')

    def save_preprocessed_file(self):
        assert self.new_data is not None, 'No preprocessing has been applied, did you call apply_preprocessing?'

        data_size = self.new_data.shape[0]
        self.data_size = (int)(data_size * (1 - self.test_split))
        self.test_size = data_size - self.data_size

        self.new_data.iloc[:self.data_size, :].to_csv(self.path + self.CLEAN_PREFIX + self.filename,
                                                      sep=self.separator,
                                                      index=False)
        self.new_data.iloc[self.data_size:, :].to_csv(self.path + self.TEST_PREFIX + self.filename, sep=self.separator,
                                                      index=False)
        print('Successfully saved preprocessed files')

    def apply_preprocessing(self, column_name):
        assert self.data is not None, 'No input data has been loaded'

        new_data = self.data.loc[self.data[column_name].str.len() < self.max_data_length].copy()
        new_data[column_name] = new_data[column_name].apply(lambda x: self.preprocess_single_entry(x))
        self._build_dictionary(new_data, column_name)

        self.new_data = new_data
        print('Applied preprocessing to input data')

    def preprocess_single_entry(self, entry):
        entry = self._regex_preprocess(entry)
        entry = self._custom_preprocessing(entry)

        return entry

    @abstractclassmethod
    def _custom_preprocessing(self, entry):
        """
        Apply custom preprocessing to single data entry. 
        :param entry: 
        :return: the entry after custom preprocessing
        """

        return entry

    def _regex_preprocess(self, entry):
        entry = self._add_space.sub(r' \1 ', entry)
        entry = self._regex.sub('', entry)
        entry = self._remove_space_after_quote.sub(r"'", entry)
        entry = self._remove_spaces.sub(' ', entry).lower().strip()

        return entry

    def read_file(self):
        self.data = pd.read_csv(self.path + self.filename, sep=self.separator)

        return self.data

    @staticmethod
    def read_vocabulary(file_path, separator):
        dictionary = pd.read_csv(file_path, sep=separator, header=None).to_dict()

        # remap value <> key to key <> value
        dictionary = {v: k for k, v in dictionary[0].items()}

        return dictionary

    @staticmethod
    def get_line_number(file_path):
        with open(file_path, "r+") as fp:
            buf = mmap.mmap(fp.fileno(), 0)
            lines = 0
            while buf.readline():
                lines += 1

        return lines
