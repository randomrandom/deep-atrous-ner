from tqdm import tqdm

from data.conll_loader import ConllLoader
from model.model import *
from preprocess import preprocess_files

__author__ = 'georgi.val.stoyan0v@gmail.com'

BATCH_SIZE = 4  # should be more than one
DEBUG_SHOW = -1  # number of prediction samples to be shown
EPOCHS = 1

BUCKETS = [1]
DATA_FILE = ['./data/datasets/conll_2003/eng.train']
VAL_FILES = ['./data/datasets/conll_2003/eng.testa']
TEST_FILES = ['./data/datasets/conll_2003/eng.testb']
OTHER_VOCABULARY_FILES = ['./data/datasets/conll_2003/vocabulary_eng.testa',
                          './data/datasets/conll_2003/vocabulary_eng.testb']

data = ConllLoader(BUCKETS, DATA_FILE, batch_size=BATCH_SIZE, use_pretrained_emb=True, used_for_test_data=True,
                   pretrained_emb_file=pre_trained_embeddings_file, other_vocabulary_files=OTHER_VOCABULARY_FILES,
                   embed_dim=embedding_dim)
validation = ConllLoader(BUCKETS, VAL_FILES, batch_size=BATCH_SIZE, table=data.table, table_pos=data.table_pos,
                         table_chunk=data.table_chunk, table_entity=data.table_entity)
test = ConllLoader(BUCKETS, TEST_FILES, batch_size=BATCH_SIZE, table=data.table, table_pos=data.table_pos,
                   table_chunk=data.table_chunk, table_entity=data.table_entity)

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

z_w = test.source_words.sg_lookup(emb=word_emb)
z_p = tf.one_hot(test.source_pos - 1, depth=num_pos)
z_c = tf.one_hot(test.source_chunk - 1, depth=num_chunk)
z_cap = test.source_capitals.sg_cast(dtype=tf.float32)

# we concatenated all inputs into one single input vector
z_i = tf.concat([z_w, z_p, z_c, z_cap], 2)

with tf.sg_context(name='model'):
    # classifier = rnn_classify(z_i, num_labels, is_test=True)
    classifier = acnn_classify(z_i, num_labels, test=True)

    # calculating precision, recall and f-1 score (more relevant than accuracy)
    predictions = classifier.sg_argmax() + 1

    words = data.reverse_table.lookup(test.source_words)
    entities = data.reverse_table_entity.lookup(test.entities)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    # init session vars
    tf.sg_init(sess)
    sess.run(tf.tables_initializer())

    tf.sg_restore(sess, 'asset/train' + max_model_name)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    try:
        all_true = []
        all_predicted = []
        for i in tqdm(range(0, EPOCHS * test.num_batches)):
            words_sample, word_ids, word_entities_sample, entities_sample, predictions_sample = sess.run(
                [words, test.source_words, entities, test.entities, predictions])

            all_true.extend(entities_sample.flatten())
            all_predicted.extend(predictions_sample.flatten())

            if i < DEBUG_SHOW:
                print('\nExample:')
                print(words_sample)
                print (word_ids)
                print(word_entities_sample)
                print(entities_sample)
                print('Predictions:')
                print(predictions_sample)

        f1_separate_scores, f1_stat, precision_separate_scores, precision_score, recall_separate_scores, recall_score = \
            calculate_f1_metrics(all_predicted, all_true)

        print('Precision scores of the meaningful classes: {}'.format(precision_separate_scores))
        print('Recall scores of the meaningful classes: {}'.format(recall_separate_scores))
        print('F1 scores of the meaningful classes: {}'.format(f1_separate_scores))
        print('Total precision score: {}'.format(precision_score))
        print('Total recall score: {}'.format(recall_score))
        print('Total f1 score: {}'.format(f1_stat))
    except tf.errors.OutOfRangeError as ex:
        coord.request_stop(ex=ex)
    finally:
        coord.request_stop()
        coord.join(threads)
