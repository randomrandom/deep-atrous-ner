from data.conll_loader import ConllLoader
from model.model import *

__author__ = 'georgi.val.stoyan0v@gmail.com'

BATCH_SIZE = 1

BUCKETS = [5, 10, 15, 20, 30]
DATA_FILE = ['./data/datasets/conll_2003/eng.train']

data = ConllLoader(BUCKETS, DATA_FILE, used_for_test_data=True, batch_size=BATCH_SIZE)

words = tf.placeholder(dtype=tf.string, shape=BATCH_SIZE)
pos = tf.placeholder(dtype=tf.string, shape=BATCH_SIZE)
chunks = tf.placeholder(dtype=tf.string, shape=BATCH_SIZE)
capitals = tf.placeholder(dtype=tf.string, shape=BATCH_SIZE)

# preprocess things
p_words, p_pos, p_chunks, p_capitals = data.build_eval_graph(words, pos, chunks, capitals)

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
    # entities_emb = tf.sg_emb(name='entities_emb', voca_size=num_labels, dim=2)

# data.visualize_embeddings(sess, word_emb, word_embedding_name)

with tf.sg_context(name='model'):
    z_w = p_words.sg_lookup(emb=word_emb)
    z_p = p_pos.sg_lookup(emb=pos_emb)
    z_c = p_chunks.sg_lookup(emb=chunk_emb)
    # z_cap = opt.capitals[opt.gpu_index].sg_cast(dtype=tf.float32)

    # we concatenated all inputs into one single input vector
    z_i = tf.concat([z_w, z_p, z_c], 2)

    classifier = decode(z_i, num_labels, data.vocabulary_size)

score = classifier.sg_argmax(axis=2)
entities = data.reverse_table.lookup(score)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    # init session vars
    tf.sg_init(sess)
    sess.run(tf.tables_initializer())
    tf.sg_restore(sess, tf.train.latest_checkpoint('asset/train'))

    exit_command = 'quit'
    print('Enter an example or write \'%s\' to exit' % exit_command)

    while True:
        i_sentence = input("Enter your a sentence: ")
        if i_sentence == exit_command: break

        i_pos = input("Enter the PoS tags: ")
        i_chunks = input("Enter the CHUNK tags: ")

        i_sentence = data.process_console_input(i_sentence)
        result = sess.run(entities, {words: [i_sentence], pos: [i_pos], chunks: [i_chunks]})

        print('Tags: %s' % str(result))
