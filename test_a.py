from tqdm import tqdm

from data.conll_loader import ConllLoader
from model.model import *

__author__ = 'georgi.val.stoyan0v@gmail.com'

BATCH_SIZE = 250

BUCKETS = [5, 10, 15, 20, 30]
DATA_FILE = ['./data/datasets/conll_2003/eng.train']
TEST_FILES = ['./data/datasets/conll_2003/eng.testa']
NUM_LABELS = 9

data = ConllLoader(BUCKETS, DATA_FILE, batch_size=BATCH_SIZE)
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
    pos_emb = tf.sg_emb(name='pos_emb', voca_size=46, dim=5)
    chunk_emb = tf.sg_emb(name='chunk_emb', voca_size=18, dim=2)

with tf.sg_context(name='model'):
    z_w = test.source_words.sg_lookup(emb=word_emb)
    z_p = test.source_pos.sg_lookup(emb=pos_emb)
    z_c = test.source_chunk.sg_lookup(emb=chunk_emb)
    z_cap = test.source_capitals.sg_cast(dtype=tf.float32)

    # we concatenated all inputs into one single input vector
    z_i = tf.concat([z_w, z_p, z_c, z_cap], 2)

    classifier = decode(z_i, NUM_LABELS, data.vocabulary_size)

    # calculating precision, recall and f-1 score (more relevant than accuracy)
    predictions = classifier.sg_argmax(axis=2)
    precision, precision_op = tf.contrib.metrics.streaming_precision(test.entities, predictions)
    recall, recall_op = tf.contrib.metrics.streaming_recall(test.entities, predictions)

    f1_score = (2 * (precision_op * recall_op)) / (precision_op + recall_op)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    # init session vars
    tf.sg_init(sess)
    sess.run(tf.tables_initializer())
    tf.sg_restore(sess, tf.train.latest_checkpoint('asset/train'))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        for i in tqdm(range(0, test.data_size // BATCH_SIZE)):
            sess.run([precision_op, recall_op])

        final_precision, final_recall, final_f1 = sess.run([precision, recall, f1_score])
        print('Precision:{}'.format(final_precision))
        print('Recall:{}'.format(final_recall))
        print('f-1 score:{}'.format(final_f1))
    except tf.errors.OutOfRangeError as ex:
        coord.request_stop(ex=ex)
    finally:
        coord.request_stop()
        coord.join(threads)
