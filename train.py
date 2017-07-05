from data.conll_loader import ConllLoader
from model.model import *
from model.trainer import classifier_train

__author__ = 'georgi.val.stoyan0v@gmail.com'

BATCH_SIZE = 16

BUCKETS = [5, 10, 15, 20, 30]
DATA_FILE = ['data/datasets/conll_2003/eng.train']
NUM_LABELS = 9

data = ConllLoader(BUCKETS, DATA_FILE, batch_size=BATCH_SIZE)
# validation = KaggleLoader(BUCKETS, DATA_FILE, used_for_test_data=True, batch_size=BATCH_SIZE)

# print(data.source_words)
words, pos = tf.split(data.source_words, tf.sg_gpus()), tf.split(data.source_pos, tf.sg_gpus())
chunks, capitals = tf.split(data.source_chunk, tf.sg_gpus()), tf.split(data.source_capitals, tf.sg_gpus())
entities = tf.split(data.entities, tf.sg_gpus())

# val_x, val_y = tf.split(validation.source, tf.sg_gpus()), tf.split(validation.target, tf.sg_gpus())

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
        z_e = opt.entities[opt.gpu_index]#.sg_lookup(emb=entities_emb)

        # we concatenated all inputs into one single input vector
        z_i = tf.concat([z_w, z_p, z_c], 2)

        train_classifier = decode(z_i, NUM_LABELS, data.vocabulary_size)

        # cross entropy loss with logit
        loss = train_classifier.sg_ce(target=z_e)

        return loss

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

'''
@tf.sg_parallel
def get_val_metrics(opt):
    with tf.sg_context(name='model', reuse=True):
        tf.get_variable_scope().reuse_variables()

        v_x = opt.input[opt.gpu_index].sg_lookup(emb=emb)

        test_classifier = decode(v_x, NUM_LABELS, validation.vocabulary_size)

        # accuracy evaluation (validation set)
        acc = (test_classifier.sg_softmax()
               .sg_accuracy(target=opt.target[opt.gpu_index], name='accuracy'))

        # validation loss
        val_loss = (test_classifier.sg_ce(target=opt.target[opt.gpu_index], name='validation'))

        return acc, val_loss
'''

# train
classifier_train(sess=sess, log_interval=50, lr=1e-3,
                 loss=get_train_loss(words=words, pos=pos, chunks=chunks, capitals=capitals, entities=entities)[0],
                 # eval_metric=get_val_metrics(input=val_x, target=val_y)[0],
                 ep_size=data.num_batches, max_ep=10, early_stop=False)
