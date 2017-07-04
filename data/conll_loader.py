import sugartensor as tf

from data.base_data_loader import BaseDataLoader

__author__ = 'georgi.val.stoyan0v@gmail.com'


class ConllLoader(BaseDataLoader):
    _name = 'ConllLoader'
    TSV_DELIM = ' '
    DATA_COLUMN = 'EXAMPLPE'

    DEFAULT_META_DATA_FILE = 'metadata_labeledTrainData.tsv'
    DEFAULT_METADATA_DIR = 'data/datasets/conll_2003/'

    def __init__(self, bucket_boundaries, file_names, *args, **kwargs):
        self.__file_preprocessor = None

        self.field_delim = ConllLoader.TSV_DELIM
        self.file_names = file_names

        record_defaults = [[""], [""], [""], [""]]
        skip_header_lines = 1
        data_column = ConllLoader.DATA_COLUMN

        super().__init__(record_defaults, self.field_delim, data_column, bucket_boundaries, file_names, *args,
                         skip_header_lines=skip_header_lines, meta_file=ConllLoader.DEFAULT_META_DATA_FILE,
                         save_dir=ConllLoader.DEFAULT_METADATA_DIR, **kwargs)

        self.source, self.target = self.get_data()

        data_size = self.data_size
        self.num_batches = data_size // self._batch_size

    def _read_file(self, filename_queue, record_defaults, field_delim=BaseDataLoader._CSV_DELIM,
                   skip_header_lines=BaseDataLoader._DEFAULT_SKIP_HEADER_LINES):
        """
        Reading of the ConLL TSV file
        :param filename_queue: 
        :return: single example and label
        """

        reader = tf.TextLineReader(skip_header_lines=skip_header_lines)
        key, value = reader.read(filename_queue)

        def is_good_example(word):
            return word != '' and word != ConllLoader.DATA_COLUMN

        inside_example=False

        all_words, all_pos, all_chunk, all_entity = [], [], [], []
        while True:
            word, pos, chunk, entity = tf.decode_csv(value, record_defaults, field_delim)

            if is_good_example(word):
                if not inside_example:
                    inside_example = True

                all_words.append(word.lower())
                all_pos.append(pos)
                all_chunk.append(chunk)
                all_entity.append(entity)

            elif inside_example:
                # we are no longer within a valid example, but our state tells us that we have entered one already
                # => means that the example has ended
                break

        return ' '.join(all_words), ' '.join(all_pos), ' '.join(all_chunk), ' '.join(all_entity)
