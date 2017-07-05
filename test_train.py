import sugartensor as tf
from model.model import *

from data.conll_loader import ConllLoader

__author__ = 'georgi.val.stoyan0v@gmail.com'

BATCH_SIZE = 8

BUCKETS = [5, 10, 15, 20]
DATA_FILE = ['data/datasets/conll_2003/eng.train']
NUM_LABELS = 9

data = ConllLoader(BUCKETS, DATA_FILE, batch_size=BATCH_SIZE)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

word_emb = tf.sg_emb(name='emb', voca_size=data.vocabulary_size, dim=embedding_dim)

with tf.sg_context(name='model'):
    z_w = data.source_words.sg_lookup(emb=word_emb)

    train_classifier = decode(z_w, NUM_LABELS)

    loss = train_classifier.sg_ce(target=data.entities)

with sess:
    initializer = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer(), tf.tables_initializer())
    sess.run(initializer)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        print(sess.run([data.source_words, data.source_pos]))#, data.source_pos, data.source_chunk, data.source_capitals]))
        #print(sess.run([z_w]))
        #print(sess.run([train_classifier]))
        #print(sess.run([loss]))

    except tf.errors.OutOfRangeError as ex:
        coord.request_stop(ex=ex)
    finally:
        coord.request_stop()
        coord.join(threads)
