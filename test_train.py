from sklearn import metrics
from tqdm import tqdm

from data.conll_loader import ConllLoader
from model.model import *

__author__ = 'georgi.val.stoyan0v@gmail.com'

BATCH_SIZE = 128

BUCKETS = [20, 40, 80, 120, 180]
DATA_FILE = ['./data/datasets/conll_2003/eng.train']
TEST_FILES = ['./data/datasets/conll_2003/eng.testa']

data = ConllLoader(BUCKETS, DATA_FILE, batch_size=BATCH_SIZE)
validation = ConllLoader(BUCKETS, TEST_FILES, batch_size=BATCH_SIZE, table=data.table, table_pos=data.table_pos,
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
    embedding_matrix = data.preload_embeddings(embedding_dim, pre_trained_embeddings_file)
    word_emb = init_custom_embeddings(name=word_embedding_name, embeddings_matrix=embedding_matrix, trainable=False)
else:
    word_emb = tf.sg_emb(name=word_embedding_name, voca_size=data.vocabulary_size, dim=embedding_dim)

z_w = data.source_words.sg_lookup(emb=word_emb)
z_p = tf.one_hot(data.source_pos, depth=num_pos)
z_c = tf.one_hot(data.source_chunk, depth=num_chunk)
z_cap = data.source_capitals.sg_cast(dtype=tf.float32)

# we concatenated all inputs into one single input vector
z_i = tf.concat([z_w, z_p, z_c, z_cap], 2)  # tf.split(tf.concat([z_w, z_p, z_c, z_cap], 2), tf.sg_gpus())

v_w = validation.source_words.sg_lookup(emb=word_emb)
v_p = tf.one_hot(validation.source_pos, depth=num_pos)
v_c = tf.one_hot(validation.source_chunk, depth=num_chunk)
v_cap = validation.source_capitals.sg_cast(dtype=tf.float32)

# we concatenated all inputs into one single input vector
v_i = tf.concat([v_w, v_p, v_c, v_cap], 2)  # tf.split(tf.concat([v_w, v_p, v_c, v_cap], 2), tf.sg_gpus())

entities = data.entities  # tf.split(data.entities, tf.sg_gpus())
val_entities = validation.entities  # tf.split(validation.entities, tf.sg_gpus())

# setup the model for training and validation. Enable multi-GPU support
with tf.sg_context(name='model'):
    labels = entities

    # train_classifier = rnn_classify(opt.z_i[opt.gpu_index], num_labels)
    train_classifier = acnn_classify(z_i, num_labels)

    # cross entropy loss with logit
    loss = train_classifier.ner_cost(target=labels, num_classes=num_labels)
    # loss = train_classifier.sg_ce(target=labels, mask=True)
    # loss += 5 * tf.not_equal(labels, tf.ones_like(labels)).sg_float()
    # loss *= tf.not_equal(labels, tf.zeros_like(labels)).sg_float()

with tf.sg_context(name='model', reuse=True):
    tf.get_variable_scope().reuse_variables()

    labels = val_entities

    # test_classifier = rnn_classify(opt.v_i[opt.gpu_index], num_labels, is_test=True)
    test_classifier = acnn_classify(v_i, num_labels, test=True)

    # accuracy evaluation (validation set)
    acc = test_classifier.ner_accuracy(target=labels, mask=True, name='accuracy')

    # calculating precision, recall and f-1 score (more relevant than accuracy)
    predictions = test_classifier.sg_argmax()
    one_hot_predictions = tf.one_hot(predictions, num_labels, dtype=tf.float64)
    one_hot_labels = tf.one_hot(labels - 1, num_labels, dtype=tf.int64)

    weights = tf.not_equal(labels, tf.zeros_like(labels)).sg_float()
    precision, precision_op = tf.contrib.metrics.streaming_sparse_average_precision_at_k(one_hot_predictions,
                                                                                         one_hot_labels, 1,
                                                                                         weights=weights,
                                                                                         name='val_precision')

    recall, recall_op = tf.contrib.metrics.streaming_sparse_recall_at_k(one_hot_predictions, one_hot_labels, 1,
                                                                        weights=weights, name='val_recall')

    f1_score = (2 * (precision_op * recall_op)) / (precision_op + recall_op)

    val_lengths = tf.cast(tf.reduce_sum(tf.sign(labels), reduction_indices=1), tf.int32)

    # validation loss
    val_loss = test_classifier.ner_cost(target=labels, mask=True, num_classes=num_labels, name='val_loss')
    # val_loss = test_classifier.sg_ce(target=labels, mask=True, name='val_loss')

tf.sg_init(sess)
data.visualize_embeddings(sess, word_emb, word_embedding_name)


def f1(class_size, prediction, target, length):
    tp = np.array([0] * (class_size + 1))
    fp = np.array([0] * (class_size + 1))
    fn = np.array([0] * (class_size + 1))

    for i in range(len(target)):
        for j in range(length[i]):
            if target[i][j] == prediction[i][j]:
                tp[target[i][j]] += 1
            else:
                fp[target[i][j]] += 1
                fn[prediction[i][j]] += 1
    unnamed_entity = 1  # the O sign
    for i in range(2, class_size):
        if i != unnamed_entity:
            tp[class_size] += tp[i]
            fp[class_size] += fp[i]
            fn[class_size] += fn[i]
    precision = []
    recall = []
    fscore = []
    for i in range(class_size + 1):
        precision.append(tp[i] * 1.0 / (tp[i] + fp[i]))
        recall.append(tp[i] * 1.0 / (tp[i] + fn[i]))
        fscore.append(2.0 * precision[i] * recall[i] / (precision[i] + recall[i]))
    print(fscore)
    return fscore[class_size]


maximum = 0
optimizer = tf.train.AdamOptimizer(0.003)
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 10)
train_op = optimizer.apply_gradients(zip(grads, tvars))

with sess:
    # init session vars
    tf.sg_init(sess)
    sess.run(tf.tables_initializer())
    # tf.sg_restore(sess, tf.train.latest_checkpoint('asset/train'))
    saver = tf.train.Saver()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        for e in range(50):
            for ptr in tqdm(range(0, data.num_batches)):
                sess.run(train_op)
            if e % 10 == 0:
                save_path = saver.save(sess, "model.ckpt")
                print("model saved in file: %s" % save_path)
            all_pred, all_lengths, all_targets = [], [], []
            for ptr in tqdm(range(0, validation.num_batches)):
                pred_, lengths_, ent_, _, __ = sess.run(
                    [test_classifier.sg_argmax() + 1, val_lengths, val_entities, precision_op, recall_op])
                all_pred.extend(pred_)
                all_lengths.extend(lengths_)
                all_targets.extend(ent_)

            prec, rec, f1_s = sess.run([precision, recall, f1_score])

            first_class = 2
            all_true, all_predicted = [], []

            for i in range(len(all_targets)):
                for j in range(len(all_targets[i])):
                    if all_targets[i][j] > 0:
                        all_true.append(all_targets[i][j])
                        all_predicted.append(all_pred[i][j])

            class_count = num_labels + 1
            s_prec = metrics.precision_score(all_true, all_predicted,
                                             labels=[i for i in range(first_class, class_count)],
                                             average=None)
            s_prec_stat = metrics.precision_score(all_true, all_predicted,
                                                  labels=[i for i in range(first_class, class_count)], average='micro')
            s_rec = metrics.recall_score(all_true, all_predicted, labels=[i for i in range(first_class, class_count)],
                                         average=None)
            s_rec_stat = metrics.recall_score(all_true, all_predicted,
                                              labels=[i for i in range(first_class, class_count)],
                                              average='micro')
            s_f1 = metrics.f1_score(all_true, all_predicted, labels=[i for i in range(first_class, class_count)],
                                    average=None)
            s_f1_stat = metrics.f1_score(all_true, all_predicted, labels=[i for i in range(first_class, class_count)],
                                         average='micro')
            s_confusion = metrics.confusion_matrix(all_true, all_predicted)

            print(s_prec)
            print(s_prec_stat)
            print(s_rec)
            print(s_rec_stat)
            print(s_f1)
            print(s_f1_stat)
            print(s_confusion)

            print("epoch %d:" % e)
            print(prec)
            print(rec)
            print(f1_s)

            print('test_a score:')
            m = f1(num_labels, all_pred, all_targets, all_lengths)
    except tf.errors.OutOfRangeError as ex:
        coord.request_stop(ex=ex)
    finally:
        coord.request_stop()
        coord.join(threads)
