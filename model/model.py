import numpy as np
import sugartensor as tf
from sklearn import metrics

__author__ = 'georgi.val.stoyan0v@gmail.com'

#
# hyper parameters
#

EMBEDDINGS_DIR = 'model/embeddings/'
GLOVE_6B_50d_EMBEDDINGS = 'glove.6B.50d.txt'
GLOVE_6B_100d_EMBEDDINGS = 'glove.6B.100d.txt'
GLOVE_6B_200d_EMBEDDINGS = 'glove.6B.200d.txt'
GLOVE_6B_300d_EMBEDDINGS = 'glove.6B.300d.txt'
GLOVE_42B_300d_EMBEDDINGS = 'glove.42B.300d.txt'

embedding_dim = 300  # 300 # embedding dimension
latent_dim = 256  # 256 # hidden layer dimension
num_blocks = 2  # 2 # dilated blocks
reg_type = 'l2'  # type of regularization used
default_dout = 0.15  # define the default dropout rate

use_pre_trained_embeddings = True  # whether to use pre-trained embedding vectors
pre_trained_embeddings_file = EMBEDDINGS_DIR + GLOVE_6B_300d_EMBEDDINGS  # the location of the pre-trained embeddings

first_meaningful_entity = 2  # filter PAD and O entity
num_labels = 5  # the number of valid entity classes
num_pos = 5  # number of valid pos tags
num_chunk = 5  # number of chunk tags

max_model_name = '/max_model.ckpt'  # name of the checkpoint with the best model


@tf.sg_layer_func
def identity(tensor, opt):
    r"""Returns the input tensor itself.

    Args:
      tensor: A `Tensor` (automatically passed by decorator).
      opt:
        bn: Boolean. If True, batch normalization is applied.
        ln: Boolean. If True, layer normalization is applied.
        dout: A float of range [0, 100). A dropout rate. Default is 0.
        act: A name of activation function. e.g., `sigmoid`, `tanh`, etc.
    Returns:
      The same tensor as `tensor`.
    """
    return tensor


# inject the custom identity function
tf.sg_inject_func(identity)


# residual block
@tf.sg_sugar_func
def sg_res_block(tensor, opt):
    # default rate
    opt += tf.sg_opt(size=3, rate=1, causal=False, is_first=False, dout=0)

    # input dimension
    in_dim = tensor.get_shape().as_list()[-1]

    with tf.sg_context(name='block_%d_%d' % (opt.block, opt.rate)):
        # reduce dimension
        input_ = (tensor
                  .sg_bypass(act='relu', bn=(not opt.is_first), name='bypass')  # do not
                  .sg_conv1d(size=1, dim=in_dim / 2, act='relu', bn=True, regularizer=reg_type, name='conv_in'))

        # 1xk conv dilated
        out = (input_
               .sg_aconv1d(size=opt.size, rate=opt.rate, dout=opt.dout, causal=opt.causal, act='relu', bn=True,
                           regularizer=reg_type, name='aconv'))

        # dimension recover and residual connection
        out = out.sg_conv1d(size=1, dim=in_dim, regularizer=reg_type, name='conv_out') + tensor

        out = out.identity(ln=True, dout=opt.dout, name='layer_norm')

    return out


# inject residual multiplicative block
tf.sg_inject_func(sg_res_block)


# cnn acnn_classify graph
def acnn_classify(x, num_classes, test=False, causal=False):
    with tf.sg_context(name='acnn_classify'):
        dropout = 0 if test else default_dout
        res = x.sg_conv1d(size=1, dim=latent_dim, ln=True, regularizer=reg_type, name='conv_input_formatter')

        # loop dilated causal conv block
        for i in range(num_blocks):
            # res = (res.sg_res_block(size=3, block=i, rates=[1, 2, 4, 8], causal=causal, dout=dropout, is_first=i == 0))
            res = (res
                   .sg_res_block(size=3, block=i, rate=1, causal=causal, dout=dropout, is_first=True)
                   .sg_res_block(size=3, block=i, rate=2, causal=causal, dout=dropout)
                   .sg_res_block(size=3, block=i, rate=4, causal=causal, dout=dropout)
                   .sg_res_block(size=3, block=i, rate=8, causal=causal, dout=dropout)
                   .sg_res_block(size=3, block=i, rate=16, causal=causal, dout=dropout))

        batch_size = res.get_shape().as_list()[0]
        in_dim = res.get_shape().as_list()[-1]

        res = res.sg_conv1d(size=1, dim=in_dim, dout=dropout, act='relu', ln=True, regularizer=reg_type,
                            name='conv_output_formatter')

        # fully convolution layer
        res = res.sg_conv1d(size=1, dim=num_classes, act='relu', ln=True, regularizer=reg_type, name='conv_final')

        # fc layers & softmax
        res = (res.sg_reshape(shape=[-1, num_classes])
               .sg_dense(dim=num_classes, name='dense_final')
               .sg_softmax()
               .sg_reshape(shape=[batch_size, -1, num_classes]))

    return res


def lstm_cell(is_test):
    dropout = 0 if is_test else default_dout
    keep_prob = 1 - dropout

    cell = tf.nn.rnn_cell.LSTMCell(latent_dim, state_is_tuple=True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)

    return cell


def rnn_classify(x, num_classes, is_test=False):
    with tf.sg_context(name='rnn_classify'):
        fw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(is_test) for _ in range(num_blocks)], state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(is_test) for _ in range(num_blocks)], state_is_tuple=True)

        words_used_in_sent = tf.sign(tf.reduce_max(tf.abs(x), reduction_indices=2))
        length = tf.cast(tf.reduce_sum(words_used_in_sent, reduction_indices=1), tf.int32)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float32, sequence_length=length)
        output = tf.concat(outputs, 2).sg_reshape(shape=[-1, 2 * latent_dim])

        prediction = output.sg_dense(dim=num_classes, name='dense')
        res = tf.reshape(prediction, [x.get_shape().as_list()[0], -1, num_classes])

    return res


@tf.sg_sugar_func
def ner_cost(tensor, opt):
    one_hot_labels = tf.one_hot(opt.target - 1, opt.num_classes, dtype=tf.float32)
    cross_entropy = one_hot_labels * tf.log(tensor)
    cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)

    mask = tf.sign(tf.reduce_max(tf.abs(one_hot_labels), reduction_indices=2))

    cross_entropy *= tf.cast(mask, tf.float32)
    cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)

    length = tf.cast(tf.reduce_sum(tf.sign(opt.target), reduction_indices=1), tf.int32)
    cross_entropy /= tf.cast(length, tf.float32)

    out = tf.reduce_mean(cross_entropy, name='ner_cost')

    # add summary
    tf.sg_summary_loss(out, name=opt.name)

    return out


tf.sg_inject_func(ner_cost)


def calculate_f1_metrics(all_predicted, all_targets):
    first_class = first_meaningful_entity
    class_count = len(set(all_targets))
    filtered_true, filtered_predicted = [], []

    for i in range(len(all_targets)):
        if all_targets[i] > 0:
            filtered_true.append(all_targets[i])
            filtered_predicted.append(all_predicted[i])

    precision_separate_scores = metrics.precision_score(filtered_true, filtered_predicted,
                                                        labels=[i for i in range(first_class, class_count)],
                                                        average=None)
    precision_score = metrics.precision_score(filtered_true, filtered_predicted,
                                              labels=[i for i in range(first_class, class_count)], average='micro')
    recall_separate_scores = metrics.recall_score(filtered_true, filtered_predicted,
                                                  labels=[i for i in range(first_class, class_count)], average=None)
    recall_score = metrics.recall_score(filtered_true, filtered_predicted,
                                        labels=[i for i in range(first_class, class_count)], average='micro')
    f1_separate_scores = metrics.f1_score(filtered_true, filtered_predicted,
                                          labels=[i for i in range(first_class, class_count)], average=None)
    f1_score = metrics.f1_score(filtered_true, filtered_predicted,
                                labels=[i for i in range(first_class, class_count)], average='micro')

    return f1_separate_scores, f1_score, precision_separate_scores, precision_score, recall_separate_scores, recall_score


@tf.sg_sugar_func
def ner_accuracy(tensor, opt):
    r"""Returns accuracy of predictions.

    Args:
      tensor: A `Tensor`. Probability distributions or unscaled prediction scores.
      opt:
        target: A 'Tensor`. Labels.

    Returns:
      A `Tensor` of the same shape as `tensor`. Each value will be 1 if correct else 0. 

    For example,

    ```
    tensor = [[20.1, 18, -4.2], [0.04, 21.1, 31.3]]
    target = [[0, 1]]
    tensor.sg_accuracy(target=target) => [[ 1.  0.]]
    ```
    """
    assert opt.target is not None, 'target is mandatory.'
    opt += tf.sg_opt(k=1)

    # # calc accuracy
    out = tf.identity(tf.equal(tensor.sg_argmax() + 1, tf.cast(opt.target, tf.int64)).sg_float(), name='acc')
    # out = tf.identity(tf.nn.in_top_k(tensor, opt.target, opt.k).sg_float(), name='acc')

    # masking padding
    if opt.mask:
        out += tf.equal(opt.target, tf.zeros_like(opt.target)).sg_float()

    return out


tf.sg_inject_func(ner_accuracy)


def init_custom_embeddings(name, embeddings_matrix, summary=True, trainable=False):
    """
    Initializes the embedding vector with custom preloaded embeddings
    """

    embedding = np.array(embeddings_matrix)
    emb = tf.get_variable(name, shape=embedding.shape, initializer=tf.constant_initializer(embedding),
                          trainable=trainable)

    if summary:
        tf.sg_summary_param(emb)

    return emb
