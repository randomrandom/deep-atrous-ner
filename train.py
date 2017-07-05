from data.conll_loader import ConllLoader
from model.model import *
from model.trainer import classifier_train

__author__ = 'georgi.val.stoyan0v@gmail.com'

BATCH_SIZE = 16

BUCKETS = [5, 10, 15, 20, 30]
DATA_FILE = ['./data/datasets/conll_2003/eng.train']
TEST_FILES = ['./data/datasets/conll_2003/eng.testa']
NUM_LABELS = 9

data = ConllLoader(BUCKETS, DATA_FILE, batch_size=BATCH_SIZE)
validation = ConllLoader(BUCKETS, TEST_FILES, batch_size=BATCH_SIZE, table=data.table, table_pos=data.table_pos,
                         table_chunk=data.table_chunk, table_entity=data.table_entity)

words, pos = tf.split(data.source_words, tf.sg_gpus()), tf.split(data.source_pos, tf.sg_gpus())
chunks, capitals = tf.split(data.source_chunk, tf.sg_gpus()), tf.split(data.source_capitals, tf.sg_gpus())
entities = tf.split(data.entities, tf.sg_gpus())

val_words, val_pos = tf.split(validation.source_words, tf.sg_gpus()), tf.split(validation.source_pos, tf.sg_gpus())
val_chunks, val_capitals = tf.split(validation.source_chunk, tf.sg_gpus()), tf.split(validation.source_capitals,
                                                                                     tf.sg_gpus())
val_entities = tf.split(validation.entities, tf.sg_gpus())

# session with multiple GPU support
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

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
    # entities_emb = tf.sg_emb(name='entities_emb', voca_size=NUM_LABELS, dim=2)


# data.visualize_embeddings(sess, word_emb, word_embedding_name)

# setup the model for training and validation. Enable multi-GPU support
@tf.sg_parallel
def get_train_loss(opt):
    with tf.sg_context(name='model'):
        z_w = opt.words[opt.gpu_index].sg_lookup(emb=word_emb)
        z_p = opt.pos[opt.gpu_index].sg_lookup(emb=pos_emb)
        z_c = opt.chunks[opt.gpu_index].sg_lookup(emb=chunk_emb)
        # z_cap = opt.capitals[opt.gpu_index].sg_cast(dtype=tf.float32)

        # we concatenated all inputs into one single input vector
        z_i = tf.concat([z_w, z_p, z_c], 2)

        train_classifier = decode(z_i, NUM_LABELS, data.vocabulary_size)

        # cross entropy loss with logit
        loss = train_classifier.sg_ce(target=opt.entities[opt.gpu_index])

        return loss


@tf.sg_parallel
def get_val_metrics(opt):
    with tf.sg_context(name='model', reuse=True):
        tf.get_variable_scope().reuse_variables()

        v_w = opt.words[opt.gpu_index].sg_lookup(emb=word_emb)
        v_p = opt.pos[opt.gpu_index].sg_lookup(emb=pos_emb)
        v_c = opt.chunks[opt.gpu_index].sg_lookup(emb=chunk_emb)
        # v_cap = opt.capitals[opt.gpu_index].sg_cast(dtype=tf.float32)

        # we concatenated all inputs into one single input vector
        v_i = tf.concat([v_w, v_p, v_c], 2)

        test_classifier = decode(v_i, NUM_LABELS, data.vocabulary_size)

        # accuracy evaluation (validation set)
        acc = (test_classifier.sg_softmax()
               .sg_accuracy(target=opt.entities[opt.gpu_index], name='accuracy'))

        # validation loss
        val_loss = (test_classifier.sg_ce(target=opt.entities[opt.gpu_index], name='val_loss'))

        return acc, val_loss


# with tf.sg_context(name='model'):
#     z_w = data.source_words.sg_lookup(emb=word_emb)
#
#     train_classifier = decode(z_w, NUM_LABELS)
#
#     loss = train_classifier.sg_ce(target=data.entities)
#     '''
#     z_w = data.source_words.sg_lookup(emb=word_emb)
#
#     enc = encode(z_w)
#
#     # shift target for training source
#     y_in = tf.concat([tf.zeros((BATCH_SIZE, 1), tf.int64), data.entities[:, :-1]], axis=1)
#     z_y = y_in.sg_lookup(emb=entities_emb)
#
#     enc = enc.sg_concat(target=z_y)
#
#     dec = decode(enc, NUM_LABELS)
#
#     loss = dec.sg_ce(target=data.entities, mask=True)
#     '''
#

# train
classifier_train(sess=sess, log_interval=50, lr=1e-3,
                 loss=get_train_loss(words=words, pos=pos, chunks=chunks, capitals=capitals, entities=entities)[0],
                 eval_metric=get_val_metrics(words=val_words, pos=val_pos, chunks=val_chunks, capitals=val_capitals,
                                             entities=val_entities)[0],
                 ep_size=data.num_batches, max_ep=10, early_stop=False)
