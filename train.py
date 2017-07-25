from data.conll_loader import ConllLoader
from model.model import *
from model.trainer import classifier_train

__author__ = 'georgi.val.stoyan0v@gmail.com'

BATCH_SIZE = 256

BUCKETS = [20, 60, 80, 120, 180]
DATA_FILE = ['./data/datasets/conll_2003/eng.train']
VAL_FILES = ['./data/datasets/conll_2003/eng.testa']
TEST_FILES = ['./data/datasets/conll_2003/eng.testb']
OTHER_VOCABULARY_FILES = ['./data/datasets/conll_2003/vocabulary_eng_testa',
                          './data/datasets/conll_2003/vocabulary_eng_testb']

data = ConllLoader(BUCKETS, DATA_FILE, batch_size=BATCH_SIZE, use_pretrained_emb=True,
                   pretrained_emb_file=pre_trained_embeddings_file, other_vocabulary_files=OTHER_VOCABULARY_FILES,
                   embed_dim=embedding_dim)
validation = ConllLoader(BUCKETS, VAL_FILES, batch_size=BATCH_SIZE, table=data.table, table_pos=data.table_pos,
                         table_chunk=data.table_chunk, table_entity=data.table_entity)
test = ConllLoader(BUCKETS, TEST_FILES, batch_size=BATCH_SIZE, table=data.table, table_pos=data.table_pos,
                   table_chunk=data.table_chunk, table_entity=data.table_entity)

# session with multiple GPU support
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

# setup embeddings, preload pre-trained embeddings if needed
word_emb = None
pos_emb = None
chunk_emb = None
entities_emb = None
word_embedding_name = 'word_emb'

if use_pre_trained_embeddings:
    embedding_matrix = data.pretrained_emb_matrix
    word_emb = init_custom_embeddings(name=word_embedding_name, embeddings_matrix=embedding_matrix, trainable=False)
else:
    word_emb = tf.sg_emb(name=word_embedding_name, voca_size=data.vocabulary_size, dim=embedding_dim)

# train inputs
z_w = data.source_words.sg_lookup(emb=word_emb)
z_p = tf.one_hot(data.source_pos - 1, depth=num_pos)
z_c = tf.one_hot(data.source_chunk - 1, depth=num_chunk)
z_cap = data.source_capitals.sg_cast(dtype=tf.float32)

# we concatenated all inputs into one single input vector
z_i = tf.split(tf.concat([z_w, z_p, z_c, z_cap], 2), tf.sg_gpus())

# validation inputs
v_w = validation.source_words.sg_lookup(emb=word_emb)
v_p = tf.one_hot(validation.source_pos - 1, depth=num_pos)
v_c = tf.one_hot(validation.source_chunk - 1, depth=num_chunk)
v_cap = validation.source_capitals.sg_cast(dtype=tf.float32)

# we concatenated all inputs into one single input vector
v_i = tf.split(tf.concat([v_w, v_p, v_c, v_cap], 2), tf.sg_gpus())

# test inputs
t_w = test.source_words.sg_lookup(emb=word_emb)
t_p = tf.one_hot(test.source_pos - 1, depth=num_pos)
t_c = tf.one_hot(test.source_chunk - 1, depth=num_chunk)
t_cap = test.source_capitals.sg_cast(dtype=tf.float32)

# we concatenated all inputs into one single input vector
t_i = tf.split(tf.concat([t_w, t_p, t_c, t_cap], 2), tf.sg_gpus())

entities = tf.split(data.entities, tf.sg_gpus())
val_entities = tf.split(validation.entities, tf.sg_gpus())
test_entities = tf.split(test.entities, tf.sg_gpus())


# setup the model for training and validation. Enable multi-GPU support
@tf.sg_parallel
def get_train_loss(opt):
    with tf.sg_context(name='model'):
        labels = opt.entities[opt.gpu_index]

        train_classifier = acnn_classify(opt.input[opt.gpu_index], num_labels)

        # cross entropy loss with logit
        loss = train_classifier.ner_cost(target=labels, num_classes=num_labels)

        return loss


@tf.sg_parallel
def get_val_metrics(opt):
    with tf.sg_context(name='model', reuse=True):
        tf.get_variable_scope().reuse_variables()

        val_labels = opt.entities[opt.gpu_index]

        test_classifier = acnn_classify(opt.input[opt.gpu_index], num_labels, test=True)
        val_predictions = test_classifier.sg_argmax() + 1

        # accuracy evaluation (validation set)
        val_acc = test_classifier.ner_accuracy(target=val_labels, mask=True, name='accuracy')

        # validation loss
        val_loss = test_classifier.ner_cost(target=val_labels, num_classes=num_labels, name='val_loss')

        return val_acc, val_loss, val_predictions, val_labels


@tf.sg_parallel
def get_test_metrics(opt):
    with tf.sg_context(name='model', reuse=True):
        tf.get_variable_scope().reuse_variables()

        test_labels = opt.entities[opt.gpu_index]

        # test_classifier = rnn_classify(opt.v_i[opt.gpu_index], num_labels, is_test=True)
        test_classifier = acnn_classify(opt.input[opt.gpu_index], num_labels, test=True)
        test_predictions = test_classifier.sg_argmax() + 1

        return test_predictions, test_labels


tf.sg_init(sess)
data.visualize_embeddings(sess, word_emb, word_embedding_name)

# train
classifier_train(sess=sess, log_interval=30, lr=3e-3, clip_grad_norm=10, optim='Adam', max_keep=10,
                 loss=get_train_loss(input=z_i, entities=entities)[0],
                 eval_metric=get_val_metrics(input=v_i, entities=val_entities)[0], ep_size=data.num_batches,
                 test_metric=get_test_metrics(input=t_i, entities=test_entities)[0],
                 test_ep_size=test.num_batches, val_ep_size=validation.num_batches, max_ep=150, early_stop=False)
