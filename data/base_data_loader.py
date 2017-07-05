from pathlib import Path

import ntpath
import numpy as np
import sugartensor as tf
from tensorflow.contrib.tensorboard.plugins import projector

from data.preprocessors.conll_preprocessor import ConllPreprocessor

__author__ = 'george.val.stoyan0v@gmail.com'


class BaseDataLoader(object):
    _CSV_DELIM = ","
    _DEFAULT_SKIP_HEADER_LINES = 0
    _name = "data_loader"
    _num_threads = 2  # 32
    _batch_size = 16  # 64
    _min_after_dequeue = _batch_size * _num_threads
    _capacity = _min_after_dequeue + (_num_threads + 2) * _batch_size  # as recommended in tf tutorial

    _TABLE_POS = ConllPreprocessor.VOCABULARY_PREFIX + ConllPreprocessor.VOCABULARY_POS
    _TABLE_CHUNK = ConllPreprocessor.VOCABULARY_PREFIX + ConllPreprocessor.VOCABULARY_CHUNK
    _TABLE_ENTITY = ConllPreprocessor.VOCABULARY_PREFIX + ConllPreprocessor.VOCABULARY_ENTITY

    DEFAULT_TEST_SPLIT = .1
    DEFAULT_MAX_DATA_LENGTH = 2000
    DEFAULT_VOCABULARY_SIZE = 50000
    DEFAULT_PRETRAINED_EMBEDDINGS = 'data/embeddings/glove.6B.300d.txt'

    DEFAULT_META_DATA_FILE = 'metadata.tsv'
    DEFAULT_METADATA_DIR = 'asset/train/'

    def __init__(self, record_defaults, field_delim, data_column, bucket_boundaries, file_names,
                 skip_header_lines=_DEFAULT_SKIP_HEADER_LINES,
                 num_threads=_num_threads, batch_size=_batch_size, min_after_dequeue=_min_after_dequeue,
                 capacity=_capacity, used_for_test_data=False, meta_file=DEFAULT_META_DATA_FILE,
                 save_dir=DEFAULT_METADATA_DIR, table=None, name=_name):
        self.__file_names = file_names
        self.__field_delim = field_delim
        self.__record_defaults = record_defaults
        self.__skip_header_lines = skip_header_lines
        self.__data_column = data_column
        self.__bucket_boundaries = bucket_boundaries
        self.__vocabulary_file = None

        self._used_for_test_data = used_for_test_data
        self._min_after_dequeue = min_after_dequeue
        self._batch_size = batch_size
        self._capacity = capacity
        self._name = name

        self.meta_file = meta_file
        self.save_dir = save_dir
        self.table = table
        self.num_threads = num_threads
        self.vocabulary_size = 0
        self.train_size = 0
        self.test_size = 0

        self.shuffle_queue = tf.RandomShuffleQueue(capacity=self._capacity, min_after_dequeue=self._min_after_dequeue,
                                                   dtypes=[tf.int64, tf.int64, tf.int64, tf.int64, tf.int64],
                                                   shapes=None)

    def get_data(self):
        return self.__load_batch(self.__file_names, record_defaults=self.__record_defaults,
                                 field_delim=self.__field_delim, data_column=self.__data_column,
                                 bucket_boundaries=self.__bucket_boundaries, skip_header_lines=self.__skip_header_lines,
                                 num_epochs=None, shuffle=True)

    @staticmethod
    def _split_file_to_path_and_name(file_name):
        file_path, tail = ntpath.split(file_name)
        file_path += '/'

        return file_path, tail

    def __generate_preprocessed_files(self, file_names, data_column, field_delim=_CSV_DELIM):
        new_file_names = []
        for filename in file_names:
            file_path, tail = BaseDataLoader._split_file_to_path_and_name(filename)

            old_file_name = tail
            prefix = ConllPreprocessor.TEST_PREFIX if self._used_for_test_data else ConllPreprocessor.CLEAN_PREFIX
            file_name = file_path + prefix + tail
            file = Path(file_name)
            new_file_names.append(file_name)

            if file.exists():
                try:
                    tf.os.remove(file_name)
                except OSError:
                    print("File not found %s" % file_name)

            self.__preprocess_file(file_path, old_file_name, field_delim, data_column)

        return new_file_names

    def __preprocess_file(self, path, file_name, field_delim, data_column):
        preprocessor = ConllPreprocessor(path, file_name, field_delim, self.DEFAULT_VOCABULARY_SIZE,
                                         self.DEFAULT_MAX_DATA_LENGTH)
        preprocessor.read_file()
        preprocessor.apply_preprocessing(data_column, ConllPreprocessor.POS_COLUMN, ConllPreprocessor.CHUNK_COLUMN,
                                         ConllPreprocessor.ENTITY_COLUMN)
        preprocessor.save_preprocessed_file()
        self.vocabulary_size = preprocessor.vocabulary_size
        self.data_size = preprocessor.data_size

    def __load_batch(self, file_names, record_defaults, data_column, bucket_boundaries, field_delim=_CSV_DELIM,
                     skip_header_lines=0,
                     num_epochs=None, shuffle=True):

        original_file_names = file_names[:]
        file_names = self.__generate_preprocessed_files(file_names, data_column, field_delim=field_delim)

        filename_queue = tf.train.string_input_producer(
            file_names, num_epochs=num_epochs, shuffle=shuffle
        )

        sentence, pos, chunks, capitals, entities = self._read_file(filename_queue, record_defaults, field_delim,
                                                                    skip_header_lines)

        voca_path, voca_name = BaseDataLoader._split_file_to_path_and_name(
            original_file_names[0])  # TODO: will be break with multiple filenames
        voca_name = ConllPreprocessor.VOCABULARY_PREFIX + voca_name
        self.__vocabulary_file = voca_path + voca_name

        # load look up tables that maps words to ids
        if self.table is None:
            self.table = tf.contrib.lookup.index_table_from_file(vocabulary_file=voca_path + voca_name,
                                                                 default_value=ConllPreprocessor.UNK_TOKEN_ID,
                                                                 num_oov_buckets=0)

        self._table_pos = tf.contrib.lookup.index_table_from_file(vocabulary_file=voca_path + self._TABLE_POS,
                                                                  num_oov_buckets=0)

        self._table_chunk = tf.contrib.lookup.index_table_from_file(vocabulary_file=voca_path + self._TABLE_CHUNK,
                                                                    num_oov_buckets=0)

        self._table_entity = tf.contrib.lookup.index_table_from_file(vocabulary_file=voca_path + self._TABLE_ENTITY,
                                                                     num_oov_buckets=0)

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
        dense_pos = self._table_pos.lookup(dense_pos)

        dense_chunks = tf.sparse_tensor_to_dense(split_chunks, default_value="")
        dense_chunks = self._table_chunk.lookup(dense_chunks)

        dense_capitals = tf.sparse_tensor_to_dense(split_capitals, default_value="")
        dense_capitals = tf.string_to_number(dense_capitals, out_type=tf.int64)

        dense_entities = tf.sparse_tensor_to_dense(split_entities, default_value="")
        dense_entities = self._table_entity.lookup(dense_entities)

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
                                                          num_threads=self._num_threads)

        # reshape shape into proper form after dequeue from bucket queue
        padded_sent = padded_sent.sg_reshape(shape=[self._batch_size, -1])
        padded_pos = padded_pos.sg_reshape(shape=[self._batch_size, -1])
        padded_chunk = padded_chunk.sg_reshape(shape=[self._batch_size, -1])
        padded_capitals = padded_capitals.sg_reshape(shape=[self._batch_size, -1])
        padded_entities = padded_entities.sg_reshape(shape=[self._batch_size, -1])

        return padded_sent, padded_pos, padded_chunk, padded_capitals, padded_entities

    def _read_file(self, filename_queue, record_defaults, field_delim=_CSV_DELIM,
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

    def preload_embeddings(self, embed_dim, file_name=DEFAULT_PRETRAINED_EMBEDDINGS):
        """
        Pre-loads word embeddings like word2vec and Glove
        :param embed_dim: the embedding dimension, currently should equal to the one in the original pre-trained vector
        :param file_name: the name of the pre-trained embeddings file
        :return: the loaded pre-trained embeddings
        """

        pre_trained_emb = np.random.uniform(-0.05, 0.05, (self.vocabulary_size, embed_dim))
        with open(file_name, 'r', encoding='utf-8') as emb_file:
            mapped_words = 0

            dictionary = ConllPreprocessor.read_vocabulary(self.__vocabulary_file, self.__field_delim)
            missing_words = dictionary.copy()

            for line in emb_file.readlines():
                row = line.strip().split(' ')
                word = row[0]

                # TODO: PCA should be added to support different embedding dimensions from pre-trained embeddings
                assert len(row[1:]) == embed_dim, \
                    'Embedding dimension should be same as the one in the pre-trained embeddings.'

                if word in dictionary:
                    mapped_words = mapped_words + 1
                    pre_trained_emb[dictionary[word] - 1] = row[1:]
                    del missing_words[word]

            print('Mapped words to pre-trained embeddings: %d' % mapped_words)

            # TODO: should do some updates in voca_size if mapped words are less, currently missing words are random embeddings which are not going to be trained
            # assert mapped_words == self.VOCABULARY_SIZE, 'Glove mapping should equal to the vocabulary size'

        print('Loaded pre-trained embeddings')

        return pre_trained_emb

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
        emb.metadata_path = tf.os.path.join(self.save_dir, self.meta_file)  # metadata file
        print(tf.os.path.abspath(emb.metadata_path))
        projector.visualize_embeddings(summary_writer, config)
