from data.preprocessors.conll_preprocessor import ConllPreprocessor

__author__ = 'georgi.val.stoyan0v@gmail.com'


def do_magic(path, file_name, delim, vocabulary_size, max_data_length):
    preprocessor = ConllPreprocessor(path, file_name, delim, vocabulary_size, max_data_length)
    preprocessor.read_file()
    preprocessor.apply_preprocessing(ConllPreprocessor.EXAMPLE_COLUMN, ConllPreprocessor.POS_COLUMN,
                                     ConllPreprocessor.CHUNK_COLUMN, ConllPreprocessor.ENTITY_COLUMN)
    preprocessor.save_preprocessed_file()
