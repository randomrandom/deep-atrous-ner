import csv
import mmap

import pandas as pd
from tqdm import tqdm

from data.preprocessors.base_preprocessor import BasePreprocessor


class ConllPreprocessor(BasePreprocessor):
    EXAMPLE_COLUMN = 'EXAMPLE'
    POS_COLUMN = 'POS'
    CHUNK_COLUMN = 'CHUNK'
    CAPITAL_COLUMN = 'CAPITAL'
    ENTITY_COLUMN = 'ENTITY'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _custom_preprocessing(self, entry):
        return entry.lower()

    def __read_from_raw_file(self, raw_file):
        number_of_lines = self.get_line_number(raw_file)
        with open(raw_file, 'r') as file:

            inside_example = False

            all_words, all_pos, all_chunk, all_capital, all_entity = [], [], [], [], []

            all_examples = []
            for line in tqdm(file, total=number_of_lines):
                def is_good_example(sample_line):
                    return len(sample_line.split()) > 0 and sample_line != '' and '-DOCSTART-' not in sample_line

                if is_good_example(line):
                    inside_example = True

                    word, pos, chunk, entity = line.split()

                    all_words.append(word.lower())
                    all_pos.append(self.preprocess_pos(pos))
                    all_chunk.append(self.preprocess_chunk(chunk))
                    all_capital.append(self.get_capital_feature(word))
                    all_entity.append(self.preprocess_entity(entity))

                elif inside_example:
                    if inside_example:
                        # we are no longer within a valid example, but our state tells us that we have entered one already
                        # => means that the example has ended and we add it to pandas

                        words_example = ' '.join(map(str, all_words))
                        pos_example = ' '.join(map(str, all_pos))
                        chunk_example = ' '.join(map(str, all_chunk))
                        capital_example = ' '.join(map(str, all_capital))
                        entity_example = ' '.join(map(str, all_entity))

                        #  append the row onto our data
                        all_examples.append([words_example, pos_example, chunk_example,
                                             capital_example, entity_example])

                        all_words, all_pos, all_chunk, all_capital, all_entity = [], [], [], [], []

                    inside_example = False

            self.data = pd.DataFrame(data=all_examples, columns=[self.EXAMPLE_COLUMN, self.POS_COLUMN,
                                                                 self.CHUNK_COLUMN, self.CAPITAL_COLUMN,
                                                                 self.ENTITY_COLUMN])

        return self.data

    def read_file(self):
        file_name = self.path + self.filename

        print('Reading file: {}'.format(file_name))
        self.data = self.__read_from_raw_file(file_name)

        return self.data

    def save_preprocessed_file(self):
        assert self.new_data is not None, 'No preprocessing has been applied, did you call apply_preprocessing?'

        preprocessed_file = self.path + self.CLEAN_PREFIX + self.filename
        self.new_data.to_csv(preprocessed_file, sep=self.separator, index=False, quoting=csv.QUOTE_NONE)

        print('Successfully saved preprocessed file: %s' % preprocessed_file)

    def preprocess_pos(self, tag):
        result = None

        if tag == 'NN' or tag == 'NNS':
            result = 'NN'
        elif tag == 'FW':
            result = 'FW'
        elif tag == 'NNP' or tag == 'NNPS':
            result = 'NNP'
        elif 'VB' in tag:
            result = 'VB'
        else:
            result = 'OTHER'

        return result

    def preprocess_chunk(self, tag):
        result = None

        if 'NP' in tag:
            result = 'NP'
        elif 'VP' in tag:
            result = 'VP'
        elif 'PP' in tag:
            result = 'PP'
        elif tag == 'O':
            result = 'O'
        else:
            result = 'OTHER'

        return result

    def preprocess_entity(self, entity):
        entity = entity.split('-')
        entity = entity[0] if len(entity) == 1 else entity[1]

        return entity

    def get_capital_feature(self, word):
        is_capital_word = word[0].isupper() + 1  # we add one because 0 values are used for padding

        return is_capital_word

    @staticmethod
    def get_line_number(file_path):
        with open(file_path, "r+") as fp:
            buf = mmap.mmap(fp.fileno(), 0)
            lines = 0
            while buf.readline():
                lines += 1

        return lines
