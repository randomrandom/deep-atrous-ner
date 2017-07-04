from pathlib import Path

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
        return entry

    def save_preprocessed_file(self):
        pass

    def apply_preprocessing(self, column_name=EXAMPLE_COLUMN, entity_column=ENTITY_COLUMN):
        assert self.data is not None, 'No input data has been loaded'

        self.new_data = self.data.copy()
        self._build_dictionary(self.new_data, column_name, entity_column, pos_column=self.POS_COLUMN,
                               chunk_column=self.CHUNK_COLUMN)

    def __read_from_raw_file(self, raw_file):
        number_of_lines = BasePreprocessor.get_line_number(raw_file)
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

                    is_capital_word = word[0].isupper() + 1  # we add one because 0 values are used for padding
                    all_words.append(word.lower())
                    all_pos.append(pos)
                    all_chunk.append(chunk)
                    all_capital.append(is_capital_word)
                    all_entity.append(entity)

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
        print('Reading file')

        file_name = self.path + self.filename

        processed_file = Path(self.path + self.CLEAN_PREFIX + self.filename)

        if processed_file.exists():
            print('File already exists, no need to recreate')

            self.data = pd.read_csv(processed_file, sep=self.separator)
        else:
            self.data = self.__read_from_raw_file(file_name)

        return self.data

    def save_preprocessed_file(self):
        assert self.new_data is not None, 'No preprocessing has been applied, did you call apply_preprocessing?'

        data_size = self.new_data.shape[0]
        self.data_size = len(self.data)

        preprocessed_file = self.path + self.CLEAN_PREFIX + self.filename
        self.new_data.to_csv(preprocessed_file, sep=self.separator, index=False)

        print('Successfully saved preprocessed file: %s' % preprocessed_file)
