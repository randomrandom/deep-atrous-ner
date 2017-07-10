import numpy as np
import sugartensor as tf

__author__ = 'georgi.val.stoyan0v@gmail.com'

#
# hyper parameters
#

EMBEDDINGS_DIR = 'model/embeddings/'
GLOVE_6B_50d_EMBEDDINGS = 'glove.6B.50d.txt'
GLOVE_6B_100d_EMBEDDINGS = 'glove.6B.100d.txt'
GLOVE_6B_200d_EMBEDDINGS = 'glove.6B.200d.txt'
GLOVE_6B_300d_EMBEDDINGS = 'glove.6B.300d.txt'

embedding_dim = 256  # 300 # embedding dimension
latent_dim = 128  # 256 # hidden layer dimension
num_blocks = 2  # 2 # dilated blocks
reg_type = 'l2'  # type of regularization used
default_dout = 0.5  # define the default dropout rate
use_pre_trained_embeddings = False  # whether to use pre-trained embedding vectors
pre_trained_embeddings_file = EMBEDDINGS_DIR + GLOVE_6B_300d_EMBEDDINGS  # the location of the pre-trained embeddings


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
                  .sg_bypass(act='relu', ln=(not opt.is_first), name='bypass')  # do not
                  .sg_conv1d(size=1, dim=in_dim / 2, act='relu', ln=True, regularizer=reg_type, name='conv_in'))

        # 1xk conv dilated
        out = (input_
               .sg_aconv1d(size=opt.size, rate=opt.rate, causal=opt.causal, act='relu', ln=True,
                           regularizer=reg_type, name='aconv'))

        # dimension recover and residual connection
        out = out.sg_conv1d(size=1, dim=in_dim, dout=opt.dout, regularizer=reg_type, name='conv_out') + tensor

    return out


# inject residual multiplicative block
tf.sg_inject_func(sg_res_block)

#
# encode graph ( atrous convolution )
#
def encode(x, test=False):

    with tf.sg_context(name='encoder'):
        dropout = 0 if test else default_dout
        res = x.sg_conv1d(size=1, dim=latent_dim, ln=True, regularizer=reg_type, name='encoder_reshaper')

        # loop dilated conv block
        for i in range(num_blocks):
            res = (res
                   .sg_res_block(size=3, block=i, rate=1, is_first=True)
                   .sg_res_block(size=3, block=i, rate=2,)
                   .sg_res_block(size=3, block=i, rate=4,)
                   .sg_res_block(size=3, block=i, rate=8))
#                  .sg_res_block(size=5, block=i, rate=16))

    return res


#
# cnn decode graph ( causal convolution )
#

def decode(x, num_classes, test=False, causal=False):
    with tf.sg_context(name='decode'):
        dropout = 0 if test else default_dout
        res = x.sg_conv1d(size=1, dim=latent_dim, ln=True, regularizer=reg_type, name='decompressor')

        # loop dilated causal conv block
        for i in range(num_blocks):
            res = (res
                   .sg_res_block(size=8, block=i, rate=1, causal=causal, is_first=True)
                   .sg_res_block(size=8, block=i, rate=2, causal=causal)
                   .sg_res_block(size=8, block=i, rate=4, causal=causal)
                   .sg_res_block(size=5, block=i, rate=8, causal=causal)
                   .sg_res_block(size=5, block=i, rate=16, causal=causal))

        in_dim = res.get_shape().as_list()[-1]
        res = res.sg_conv1d(size=1, dim=in_dim, dout=dropout, act='relu', ln=True, regularizer=reg_type, name='conv_dout_final')

        # final fully convolution layer for softmax
        res = res.sg_conv1d(size=1, dim=num_classes, ln=True, regularizer=reg_type, name='conv_relu_final')

    return res


def init_custom_embeddings(name, embeddings_matrix, summary=True,
                           trainable=False):
    """
    Initializes the embedding vector with custom preloaded embeddings
    """

    embedding = np.array(embeddings_matrix)
    w = tf.get_variable(name, shape=embedding.shape,
                        initializer=tf.constant_initializer(embedding),
                        trainable=trainable)

    if summary:
        tf.sg_summary_param(w)

    emb = w

    return emb
