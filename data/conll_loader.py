import sugartensor as tf

from data.base_data_loader import BaseDataLoader
from data.preprocessors.conll_preprocessor import ConllPreprocessor

__author__ = 'georgi.val.stoyan0v@gmail.com'


class ConllLoader(BaseDataLoader):
    DATA_COLUMN = ConllPreprocessor.EXAMPLE_COLUMN

    DEFAULT_META_DATA_FILE = 'metadata_eng.train'
    DEFAULT_SAVE_DIR = 'asset/train'

    def __init__(self, bucket_boundaries, file_names, *args, **kwargs):
        self._file_preprocessor = None

        self.field_delim = ConllLoader.TSV_DELIM
        self.file_names = file_names

        record_defaults = [[''], [''], [''], [''], ['']]
        skip_header_lines = 1
        data_column = ConllLoader.DATA_COLUMN

        super().__init__(record_defaults, self.field_delim, data_column, bucket_boundaries, file_names, *args,
                         skip_header_lines=skip_header_lines, meta_file=ConllLoader.DEFAULT_META_DATA_FILE,
                         save_dir=ConllLoader.DEFAULT_SAVE_DIR, **kwargs)

        self.source_words, self.source_pos, self.source_chunk, self.source_capitals, self.entities = self.get_data()
        self.num_batches = self.data_size // self._batch_size

    def build_eval_graph(self, words, pos, chunks, capitals):
        # convert to tensor of strings
        split_sentence = tf.string_split(words, " ")
        split_pos = tf.string_split(pos, ' ')
        split_chunks = tf.string_split(chunks, ' ')
        split_capitals = tf.string_split(capitals, ' ')

        # convert sparse to dense
        dense_words = tf.sparse_tensor_to_dense(split_sentence, default_value="")
        dense_pos = tf.sparse_tensor_to_dense(split_pos, default_value="")
        dense_chunks = tf.sparse_tensor_to_dense(split_chunks, default_value="")
        dense_capitals = tf.sparse_tensor_to_dense(split_capitals, default_value="")

        # do table lookup
        table_words = self.table.lookup(dense_words)
        table_pos = self.table_pos.lookup(dense_pos)
        table_chunks = self.table_chunk.lookup(dense_chunks)
        table_capitals = tf.string_to_number(dense_capitals, out_type=tf.int64)

        return table_words, table_pos, table_chunks, table_capitals

    def process_console_input(self, entry):
        if self._file_preprocessor is None:
            file_path, file_name = BaseDataLoader._split_file_to_path_and_name(
                self.file_names[0])  # TODO: will be break with multiple filenames
            file_name = ConllPreprocessor.VOCABULARY_PREFIX + file_name
            self._file_preprocessor = ConllPreprocessor(file_path, file_name, self.field_delim,
                                                        self.DEFAULT_VOCABULARY_SIZE, self.DEFAULT_MAX_DATA_LENGTH)

        entry = self._file_preprocessor.preprocess_single_entry(entry)

        return entry
