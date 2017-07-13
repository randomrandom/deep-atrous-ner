from data.conll_loader import ConllLoader
from model.model import *
from model.trainer import classifier_train

__author__ = 'georgi.val.stoyan0v@gmail.com'

BATCH_SIZE = 64

BUCKETS = [20, 40, 80, 120, 180]
DATA_FILE = ['./data/datasets/conll_2003/eng.train']
TEST_FILES = ['./data/datasets/conll_2003/eng.testa']

data = ConllLoader(BUCKETS, DATA_FILE, batch_size=BATCH_SIZE)
validation = ConllLoader(BUCKETS, TEST_FILES, batch_size=BATCH_SIZE, table=data.table, table_pos=data.table_pos,
                         table_chunk=data.table_chunk, table_entity=data.table_entity)

words, pos = tf.split(data.source_words, tf.sg_gpus()), tf.split(data.source_pos, tf.sg_gpus())
chunks, capitals = tf.split(data.source_chunk, tf.sg_gpus()), tf.split(data.source_capitals, tf.sg_gpus())
entities = tf.split(data.entities, tf.sg_gpus())
# decoder_in = tf.split(tf.concat([tf.zeros((BATCH_SIZE, 1), tf.int64), data.entities[:, :-1]], axis=1), tf.sg_gpus())

val_words, val_pos = tf.split(validation.source_words, tf.sg_gpus()), tf.split(validation.source_pos, tf.sg_gpus())
val_chunks, val_capitals = tf.split(validation.source_chunk, tf.sg_gpus()), tf.split(validation.source_capitals,
                                                                                     tf.sg_gpus())
val_entities = tf.split(validation.entities, tf.sg_gpus())
# val_decoder_in = tf.split(tf.concat([tf.zeros((BATCH_SIZE, 1), tf.int64), validation.entities[:, :-1]], axis=1),
#                           tf.sg_gpus())

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
    # pos_emb = tf.sg_emb(name='pos_emb', voca_size=46, dim=8)
    # chunk_emb = tf.sg_emb(name='chunk_emb', voca_size=18, dim=4)
    # entities_emb = tf.sg_emb(name='entities_emb', voca_size=num_labels, dim=4)


# data.visualize_embeddings(sess, word_emb, word_embedding_name)

# setup the model for training and validation. Enable multi-GPU support
@tf.sg_parallel
def get_train_loss(opt):
    with tf.sg_context(name='model'):
        z_w = opt.words[opt.gpu_index].sg_lookup(emb=word_emb)
        z_p = tf.one_hot(opt.pos[opt.gpu_index], depth=num_pos)
        z_c = tf.one_hot(opt.chunks[opt.gpu_index], depth=num_chunk)
        z_cap = opt.capitals[opt.gpu_index].sg_cast(dtype=tf.float32)

        # we concatenated all inputs into one single input vector
        z_i = tf.concat([z_w, z_p, z_c, z_cap], 2)

        labels = opt.entities[opt.gpu_index]

        train_classifier = rnn_model(z_i, num_labels)
        #train_classifier = decode(z_i, num_labels)

        # cross entropy loss with logit
        loss = train_classifier.ner_cost(target=labels, num_classes=num_labels)
        #loss = train_classifier.sg_ce(target=labels, mask=True)

        return loss


@tf.sg_parallel
def get_val_metrics(opt):
    with tf.sg_context(name='model', reuse=True):
        tf.get_variable_scope().reuse_variables()

        v_w = opt.words[opt.gpu_index].sg_lookup(emb=word_emb)
        v_p = tf.one_hot(opt.pos[opt.gpu_index], depth=num_pos)
        v_c = tf.one_hot(opt.chunks[opt.gpu_index], depth=num_chunk)
        v_cap = opt.capitals[opt.gpu_index].sg_cast(dtype=tf.float32)

        # we concatenated all inputs into one single input vector
        v_i = tf.concat([v_w, v_p, v_c, v_cap], 2)

        labels = opt.entities[opt.gpu_index]

        test_classifier = rnn_model(v_i, num_labels, is_test=True)
        #test_classifier = decode(v_i, num_labels, test=True)

        # accuracy evaluation (validation set)
        acc = test_classifier.sg_softmax().sg_accuracy(target=labels, name='accuracy')

        # calculating precision, recall and f-1 score (more relevant than accuracy)
        predictions = test_classifier.sg_argmax(axis=2)
        one_hot_predictions = tf.one_hot(predictions, num_labels, dtype=tf.float64)
        one_hot_labels = tf.one_hot(labels, num_labels, dtype=tf.int64)

        precision, precision_op = tf.contrib.metrics.streaming_sparse_average_precision_at_k(one_hot_predictions,
                                                                                             one_hot_labels, 1,
                                                                                             name='val_precision')
        recall, recall_op = tf.contrib.metrics.streaming_sparse_recall_at_k(one_hot_predictions, one_hot_labels, 1,
                                                                            name='val_recall')

        f1_score = (2 * (precision_op * recall_op)) / (precision_op + recall_op)

        # validation loss
        val_loss = test_classifier.ner_cost(target=labels, mask=True, num_classes=num_labels, name='val_loss')
        #val_loss = test_classifier.sg_ce(target=labels, mask=True, name='val_loss')

        return acc, val_loss, precision_op, recall_op, f1_score


# train
classifier_train(sess=sess, log_interval=30, lr=1e-3, clip_grad_norm=2, save_interval=150,
                 loss=get_train_loss(words=words, pos=pos, chunks=chunks, capitals=capitals, entities=entities)[0],
                 eval_metric=get_val_metrics(words=val_words, pos=val_pos, chunks=val_chunks, capitals=val_capitals,
                                             entities=val_entities)[0],
                 ep_size=data.num_batches, max_ep=150, early_stop=False)
