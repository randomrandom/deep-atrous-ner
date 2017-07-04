import sugartensor as tf

from data.conll_loader import ConllLoader

__author__ = 'georgi.val.stoyan0v@gmail.com'

BATCH_SIZE = 64

BUCKETS = [100, 170, 240, 290, 340]
DATA_FILE = ['data/datasets/conll_2003/eng.testa']
NUM_LABELS = 2

data = ConllLoader(BUCKETS, DATA_FILE, batch_size=BATCH_SIZE)

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

with sess:
    initializer = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer(), tf.tables_initializer())
    sess.run(initializer)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        print(sess.run([#data.source_words, data.source_pos, data.source_chunk, data.source_capitals, data.target,
                        data.o1, data.o2, data.o3, data.o4, data.o5]))
    except tf.errors.OutOfRangeError as ex:
        coord.request_stop(ex=ex)
    finally:
        coord.request_stop()
        coord.join(threads)
