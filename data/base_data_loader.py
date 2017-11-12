from pathlib import Path

import ntpath
import numpy as np
import sugartensor as tf
from tensorflow.contrib.tensorboard.plugins import projector

from data.datasets.conll_2003 import preprocess_conll
from data.preprocessors.base_preprocessor import BasePreprocessor
from data.preprocessors.conll_preprocessor import ConllPreprocessor

__author__ = 'george.val.stoyan0v@gmail.com'


class BaseDataLoader(object):
    __DEFAULT_DELIM = "\t"
    _DEFAULT_SKIP_HEADER_LINES = 0
    _DEFAULT_NUM_THREADS = 32
    _DEFAULT_BATCH_SIZE = 32
    _default_min_after_dequeue = _DEFAULT_BATCH_SIZE * _DEFAULT_NUM_THREADS
    # as recommended in tf tutorial
    _default_capacity = _default_min_after_dequeue + (_DEFAULT_NUM_THREADS + 2) * _DEFAULT_BATCH_SIZE

    _TABLE_POS = ConllPreprocessor.VOCABULARY_PREFIX + ConllPreprocessor.VOCABULARY_POS
    _TABLE_CHUNK = ConllPreprocessor.VOCABULARY_PREFIX + ConllPreprocessor.VOCABULARY_CHUNK
    _TABLE_ENTITY = ConllPreprocessor.VOCABULARY_PREFIX + ConllPreprocessor.VOCABULARY_ENTITY

    DEFAULT_MAX_DATA_LENGTH = 300
    DEFAULT_VOCABULARY_SIZE = 50000
    DEFAULT_PRE_TRAINED_EMBEDDINGS = 'model/embeddings/glove.6B.300d.txt'

    DEFAULT_META_DATA_FILE = 'metadata.tsv'
    DEFAULT_META_DATA_DIR = 'data/datasets/conll_2003/'
    DEFAULT_SAVE_DIR = BasePreprocessor.DEFAULT_SAVE_DIR

    def __init__(self, record_defaults, field_delim, data_column, bucket_boundaries, file_names,
                 skip_header_lines=_DEFAULT_SKIP_HEADER_LINES, num_threads=_DEFAULT_NUM_THREADS,
                 batch_size=_DEFAULT_BATCH_SIZE, used_for_test_data=False, meta_file=DEFAULT_META_DATA_FILE,
                 save_dir=DEFAULT_SAVE_DIR, table=None, table_pos=None, table_chunk=None, table_entity=None,
                 use_pretrained_emb=False, pretrained_emb_file=None, other_vocabulary_files=None, embed_dim=0):
        self.__file_names = file_names
        self.__field_delim = field_delim
        self.__record_defaults = record_defaults
        self.__skip_header_lines = skip_header_lines
        self.__data_column = data_column
        self.__bucket_boundaries = bucket_boundaries
        self.__vocabulary_file = None

        self._used_for_test_data = used_for_test_data
        self.num_threads = num_threads
        self._batch_size = batch_size
        self._min_after_dequeue = self._batch_size * self.num_threads
        self._capacity = self._min_after_dequeue + (self.num_threads + 2) * self._batch_size
        self._pretrained_emb_file = pretrained_emb_file
        self._embed_dim = embed_dim
        self._other_voca_files = other_vocabulary_files
        self._use_pretrained_emb = use_pretrained_emb

        self.meta_file = meta_file
        self.save_dir = save_dir
        self.table = table
        self.reverse_table = None
        self.reverse_table_entity = None
        self.table_chunk = table_chunk
        self.table_pos = table_pos
        self.table_entity = table_entity
        self.vocabulary_size = 0
        self.pretrained_emb_matrix = None

        self.shuffle_queue = tf.RandomShuffleQueue(capacity=self._capacity,
                                                   min_after_dequeue=self._min_after_dequeue,
                                                   dtypes=[tf.int64, tf.int64, tf.int64, tf.int64, tf.int64],
                                                   shapes=None, name='shuffle_queue')

    def get_data(self):
        return self.__load_data(self.__file_names, record_defaults=self.__record_defaults,
                                field_delim=self.__field_delim, data_column=self.__data_column,
                                bucket_boundaries=self.__bucket_boundaries, skip_header_lines=self.__skip_header_lines,
                                num_epochs=None, shuffle=True)

    @staticmethod
    def _split_file_to_path_and_name(file_name):
        file_path, tail = ntpath.split(file_name)
        file_path += '/'

        return file_path, tail

    def __generate_preprocessed_files(self, file_names, data_column, field_delim=__DEFAULT_DELIM):
        new_file_names = []
        for filename in file_names:
            file_path, tail = BaseDataLoader._split_file_to_path_and_name(filename)

            old_file_name = tail
            prefix = ConllPreprocessor.CLEAN_PREFIX
            file_name = file_path + prefix + tail
            file = Path(file_name)
            new_file_names.append(file_name)

            self.__preprocess_file(file_path, old_file_name, field_delim, data_column, file_exists=file.exists())

        return new_file_names

    def __preprocess_file(self, path, file_name, field_delim, data_column, file_exists=False):
        preprocessor = ConllPreprocessor(path, file_name, field_delim, self.DEFAULT_VOCABULARY_SIZE,
                                         self.DEFAULT_MAX_DATA_LENGTH)
        preprocessor.read_file()

        preprocessor.apply_preprocessing(data_column, ConllPreprocessor.POS_COLUMN, ConllPreprocessor.CHUNK_COLUMN,
                                         ConllPreprocessor.ENTITY_COLUMN, recreate_dictionary=not file_exists)
        if not file_exists:
            preprocessor.save_preprocessed_file()

        self.vocabulary_size = preprocessor.vocabulary_size
        self.data_size = preprocessor.data_size
        print(self.data_size)

    def __load_data(self, file_names, record_defaults, data_column, bucket_boundaries, field_delim=__DEFAULT_DELIM,
                    skip_header_lines=0,
                    num_epochs=None, shuffle=True):

        original_file_names = file_names[:]
        file_names = self.__generate_preprocessed_files(file_names, data_column, field_delim=field_delim)

        filename_queue = tf.train.string_input_producer(
            file_names, num_epochs=num_epochs, shuffle=shuffle
        )

        sentence, pos, chunks, capitals, entities = self._read_file(filename_queue, record_defaults, field_delim,
                                                                    skip_header_lines)

        voca_path, voca_suffix = BaseDataLoader._split_file_to_path_and_name(
            original_file_names[0])  # TODO: will be break with multiple filenames
        voca_name = ConllPreprocessor.VOCABULARY_PREFIX + voca_suffix
        self.__vocabulary_file = voca_path + voca_name

        # make sure the all the other vocabularies are cleaned and generated
        # before we try to build one big vocabulary.
        for other_voc in self._other_voca_files:
            preprocess_conll.preprocess_file(voca_path, other_voc.split("_")[-1], self.__field_delim,
                                             self.DEFAULT_VOCABULARY_SIZE, self.DEFAULT_MAX_DATA_LENGTH)

        # load look up tables that maps words to ids
        if self.table is None:
            print('vocabulary table is None => creating it')
            main_voca_file = voca_path + voca_name

            if self._use_pretrained_emb:
                self.pretrained_emb_matrix, vocabulary = self.preload_embeddings(embed_dim=self._embed_dim,
                                                                                 file_name=self._pretrained_emb_file,
                                                                                 train_vocabulary=main_voca_file,
                                                                                 other_vocabularies=self._other_voca_files)
                tensor_vocabulary = tf.constant(vocabulary)
                self.table = tf.contrib.lookup.index_table_from_tensor(tensor_vocabulary,
                                                                       default_value=ConllPreprocessor.UNK_TOKEN_ID,
                                                                       num_oov_buckets=0)
            else:
                self.table = tf.contrib.lookup.index_table_from_file(vocabulary_file=main_voca_file,
                                                                     default_value=ConllPreprocessor.UNK_TOKEN_ID,
                                                                     num_oov_buckets=0)

        if self.table_pos is None:
            print('vocabulary table_pos is None => creating it')
            self.table_pos = tf.contrib.lookup.index_table_from_file(
                vocabulary_file=voca_path + self._TABLE_POS + voca_suffix,
                num_oov_buckets=0)

        if self.table_chunk is None:
            print('vocabulary table_chunk is None => creating it')
            self.table_chunk = tf.contrib.lookup.index_table_from_file(
                vocabulary_file=voca_path + self._TABLE_CHUNK + voca_suffix,
                num_oov_buckets=0)

        if self.table_entity is None:
            print('vocabulary table_entity is None => creating it')
            self.table_entity = tf.contrib.lookup.index_table_from_file(
                vocabulary_file=voca_path + self._TABLE_ENTITY + voca_suffix,
                num_oov_buckets=0)

        if self._used_for_test_data:
            print('Reverse vocabulary is needed => creating it')
            self.reverse_table = tf.contrib.lookup.index_to_string_table_from_file(
                vocabulary_file=voca_path + voca_name)
            print('Reverse entity vocabulary is needed => creating it')
            self.reverse_table_entity = tf.contrib.lookup.index_to_string_table_from_file(
                vocabulary_file=voca_path + self._TABLE_ENTITY + voca_suffix)

        # convert to tensor of strings
        split_sentence = tf.string_split([sentence], " ")
        split_pos = tf.string_split([pos], ' ')
        split_chunks = tf.string_split([chunks], ' ')
        split_capitals = tf.string_split([capitals], ' ')
        split_entities = tf.string_split([entities], ' ')

        # determine lengths of sequences
        line_number = split_sentence.indices[:, 0]
        line_position = split_sentence.indices[:, 1]
        lengths = (tf.segment_max(data=line_position,
                                  segment_ids=line_number) + 1).sg_cast(dtype=tf.int32)

        # convert sparse to dense
        dense_sent = tf.sparse_tensor_to_dense(split_sentence, default_value="")
        dense_sent = self.table.lookup(dense_sent)

        dense_pos = tf.sparse_tensor_to_dense(split_pos, default_value="")
        dense_pos = self.table_pos.lookup(dense_pos)

        dense_chunks = tf.sparse_tensor_to_dense(split_chunks, default_value="")
        dense_chunks = self.table_chunk.lookup(dense_chunks)

        dense_capitals = tf.sparse_tensor_to_dense(split_capitals, default_value="")
        dense_capitals = tf.string_to_number(dense_capitals, out_type=tf.int64)

        dense_entities = tf.sparse_tensor_to_dense(split_entities, default_value="")
        dense_entities = self.table_entity.lookup(dense_entities)

        # get the enqueue op to pass to a coordinator to be run
        self.enqueue_op = self.shuffle_queue.enqueue(
            [dense_sent, dense_pos, dense_chunks, dense_capitals, dense_entities])
        dense_sent, dense_pos, dense_chunks, dense_capitals, dense_entities = self.shuffle_queue.dequeue()

        # add queue to queue runner
        self.qr = tf.train.QueueRunner(self.shuffle_queue, [self.enqueue_op] * self.num_threads)
        tf.train.queue_runner.add_queue_runner(self.qr)

        # reshape from <unknown> shape into proper form after dequeue from random shuffle queue
        # this is needed so next queue can automatically infer the shape properly
        dense_sent = dense_sent.sg_reshape(shape=[1, -1])
        dense_pos = dense_pos.sg_reshape(shape=[1, -1])
        dense_chunks = dense_chunks.sg_reshape(shape=[1, -1])
        dense_capitals = dense_capitals.sg_reshape(shape=[1, -1])
        dense_entities = dense_entities.sg_reshape(shape=[1, -1])

        _, (padded_sent, padded_pos, padded_chunk, padded_capitals, padded_entities) = \
            tf.contrib.training.bucket_by_sequence_length(lengths,
                                                          [dense_sent, dense_pos, dense_chunks, dense_capitals,
                                                           dense_entities],
                                                          batch_size=self._batch_size,
                                                          bucket_boundaries=bucket_boundaries,
                                                          dynamic_pad=True,
                                                          capacity=self._capacity,
                                                          num_threads=self.num_threads, name='bucket_queue')

        # reshape shape into proper form after dequeue from bucket queue
        padded_sent = padded_sent.sg_reshape(shape=[self._batch_size, -1])
        padded_pos = padded_pos.sg_reshape(shape=[self._batch_size, -1])
        padded_chunk = padded_chunk.sg_reshape(shape=[self._batch_size, -1])
        padded_capitals = padded_capitals.sg_reshape(shape=[self._batch_size, -1, 1])
        padded_entities = padded_entities.sg_reshape(shape=[self._batch_size, -1])

        return padded_sent, padded_pos, padded_chunk, padded_capitals, padded_entities

    @staticmethod
    def _read_file(filename_queue, record_defaults, field_delim=__DEFAULT_DELIM,
                   skip_header_lines=_DEFAULT_SKIP_HEADER_LINES):
        """
        Reading of the ConLL TSV file
        :param filename_queue: 
        :return: single example and label
        """

        reader = tf.TextLineReader(skip_header_lines=skip_header_lines)
        key, value = reader.read(filename_queue)

        words, pos, chunks, capitals, entities = tf.decode_csv(value, record_defaults, field_delim)

        return words, pos, chunks, capitals, entities

    def preload_embeddings(self, embed_dim, file_name=DEFAULT_PRE_TRAINED_EMBEDDINGS, train_vocabulary=None,
                           other_vocabularies=None):
        """
        Pre-loads word embeddings like word2vec and Glove
        :param other_vocabularies: 
        :param train_vocabulary: 
        :param embed_dim: the embedding dimension, currently should equal to the one in the original pre-trained vector
        :param file_name: the name of the pre-trained embeddings file
        :return: the loaded pre-trained embeddings
        """

        with open(file_name, 'r', encoding='utf-8') as emb_file:
            mapped_words = 0
            dictionary = ConllPreprocessor.read_vocabulary(train_vocabulary, self.__field_delim)

            for voca_file in other_vocabularies:
                dictionary = ConllPreprocessor.read_vocabulary(voca_file, self.__field_delim, dictionary=dictionary)

            self.vocabulary_size = len(dictionary)
            vocabulary = sorted(dictionary, key=dictionary.get)

            pre_trained_emb = np.random.uniform(-0.1, 0.1, (self.vocabulary_size, embed_dim))

            missing_words = dictionary.copy()
            invalid_words = 0

            for line in emb_file.readlines():
                row = line.strip().split(' ')
                word = row[0]

                # TODO: PCA should be added to support different embedding dimensions from pre-trained embeddings
                assert len(row[1:]) == embed_dim, \
                    'Embedding dimension should be same as the one in the pre-trained embeddings.'

                if word in dictionary:
                    vector = np.array([float(val) for val in row[1:]])
                    if len(vector) != embed_dim:
                        invalid_words += 1
                        continue

                    mapped_words = mapped_words + 1
                    pre_trained_emb[dictionary[word]] = vector
                    del missing_words[word]

            print('Invalid words count: %d' % invalid_words)
            print('Mapped words to pre-trained embeddings: %d' % mapped_words)

            # TODO: should do some updates in voca_size if mapped words are less, currently missing words are random embeddings which are not going to be trained
            # assert mapped_words == self.VOCABULARY_SIZE, 'Glove mapping should equal to the vocabulary size'

        pre_trained_emb[dictionary[BasePreprocessor.PAD_TOKEN]] = [0] * embed_dim

        print('Loaded pre-trained embeddings')

        return pre_trained_emb, vocabulary

    def visualize_embeddings(self, sess, tensor, name):
        """
        Visualises an embedding vector into Tensorboard

        :param sess: Tensorflow session object
        :param tensor:  The embedding tensor to be visualizd
        :param name: Name of the tensor
        """

        # make directory if not exist
        if not tf.os.path.exists(self.save_dir):
            tf.os.makedirs(self.save_dir)

        # summary writer
        summary_writer = tf.summary.FileWriter(self.save_dir, graph=tf.get_default_graph())

        # embedding visualizer
        config = projector.ProjectorConfig()
        emb = config.embeddings.add()
        emb.tensor_name = name  # tensor
        emb.metadata_path = tf.os.path.join(self.DEFAULT_META_DATA_DIR, self.meta_file)  # metadata file
        print(tf.os.path.abspath(emb.metadata_path))
        projector.visualize_embeddings(summary_writer, config)
