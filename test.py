from sklearn import metrics
from tqdm import tqdm

from data.conll_loader import ConllLoader
from model.model import *

__author__ = 'georgi.val.stoyan0v@gmail.com'

BATCH_SIZE = 1
DEBUG_SHOW = -1  # number of prediction samples to be shown
EPOCHS = 1

BUCKETS = [1]
DATA_FILE = ['./data/datasets/conll_2003/eng.train']
TEST_FILES = ['./data/datasets/conll_2003/eng.testa']

data = ConllLoader(BUCKETS, DATA_FILE, used_for_test_data=True, batch_size=BATCH_SIZE)
test = ConllLoader(BUCKETS, TEST_FILES, batch_size=BATCH_SIZE, table=data.table, table_pos=data.table_pos,
                   table_chunk=data.table_chunk, table_entity=data.table_entity)

# setup embeddings, preload pre-trained embeddings if needed
word_emb = None
pos_emb = None
chunk_emb = None
entities_emb = None
word_embedding_name = 'word_emb'

if use_pre_trained_embeddings:
    embedding_matrix = data.preload_embeddings(embedding_dim, pre_trained_embeddings_file)
    word_emb = init_custom_embeddings(name=word_embedding_name, embeddings_matrix=embedding_matrix, trainable=True)
else:
    word_emb = tf.sg_emb(name=word_embedding_name, voca_size=data.vocabulary_size, dim=embedding_dim)

with tf.sg_context(name='model'):
    z_w = test.source_words.sg_lookup(emb=word_emb)
    z_p = tf.one_hot(test.source_pos, depth=num_pos)
    z_c = tf.one_hot(test.source_chunk, depth=num_chunk)
    z_cap = test.source_capitals.sg_cast(dtype=tf.float32)

    # we concatenated all inputs into one single input vector
    z_i = tf.concat([z_w, z_p, z_c, z_cap], 2)

    classifier = rnn_model(z_i, num_labels, is_test=True)
    #classifier = decode(z_i, num_labels, test=True)

    # calculating precision, recall and f-1 score (more relevant than accuracy)
    predictions = classifier.sg_argmax(axis=2)
    words = data.reverse_table.lookup(test.source_words)
    entities = data.reverse_table_entity.lookup(predictions)
    one_hot_predictions = tf.one_hot(predictions, num_labels, dtype=tf.float64)
    one_hot_labels = tf.one_hot(test.entities, num_labels, dtype=tf.int64)

    precision, precision_op = tf.contrib.metrics.streaming_sparse_average_precision_at_k(one_hot_predictions,
                                                                                         one_hot_labels, 1,
                                                                                         name='test_b_precision')
    recall, recall_op = tf.contrib.metrics.streaming_sparse_recall_at_k(one_hot_predictions, one_hot_labels, 1,
                                                                        name='test_b_recall')

    f1_score = (2 * (precision_op * recall_op)) / (precision_op + recall_op)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    # init session vars
    tf.sg_init(sess)
    sess.run(tf.tables_initializer())
    tf.sg_restore(sess, tf.train.latest_checkpoint('asset/train'))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        all_true = []
        all_predicted = []
        for i in tqdm(range(0, EPOCHS * test.data_size // BATCH_SIZE)):
            words_sample, word_entities_sample, entities_sample, predictions_sample, _, __ = sess.run(
                [words, entities, test.entities, predictions, precision_op, recall_op])

            all_true.extend(entities_sample.flatten())
            all_predicted.extend(predictions_sample.flatten())

            if i < DEBUG_SHOW:
                print('\nExample:')
                print(words_sample)
                print(word_entities_sample)
                print(entities_sample)
                print('Predictions:')
                print(predictions_sample)

        first_class = 1
        s_prec = metrics.precision_score(all_true, all_predicted, labels=[i for i in range(first_class, num_labels)],
                                         average=None)
        s_prec_stat = metrics.precision_score(all_true, all_predicted,
                                              labels=[i for i in range(first_class, num_labels)], average='micro')
        s_rec = metrics.recall_score(all_true, all_predicted, labels=[i for i in range(first_class, num_labels)],
                                     average=None)
        s_rec_stat = metrics.recall_score(all_true, all_predicted, labels=[i for i in range(first_class, num_labels)],
                                          average='micro')
        s_f1 = metrics.f1_score(all_true, all_predicted, labels=[i for i in range(first_class, num_labels)],
                                average=None)
        s_f1_stat = metrics.f1_score(all_true, all_predicted, labels=[i for i in range(first_class, num_labels)],
                                     average='micro')
        s_confusion = metrics.confusion_matrix(all_true, all_predicted)

        print(s_prec)
        print(s_prec_stat)
        print(s_rec)
        print(s_rec_stat)
        print(s_f1)
        print(s_f1_stat)
        print(s_confusion)

        final_precision, final_recall, final_f1 = sess.run([precision, recall, f1_score])
        print('Precision:{}'.format(final_precision))
        print('Recall:{}'.format(final_recall))
        print('f-1 score:{}'.format(final_f1))
    except tf.errors.OutOfRangeError as ex:
        coord.request_stop(ex=ex)
    finally:
        coord.request_stop()
        coord.join(threads)
