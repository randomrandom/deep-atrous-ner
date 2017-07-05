from data.conll_loader import ConllLoader
from data.datasets.conll_2003 import preprocess_conll

__author__ = 'georgi.val.stoyan0v@gmail.com'

# preprocess_conll.do_magic('./data/datasets/conll_2003/', 'eng.testa', ConllLoader.TSV_DELIM,
#                          ConllLoader.DEFAULT_VOCABULARY_SIZE, ConllLoader.DEFAULT_MAX_DATA_LENGTH)

preprocess_conll.do_magic('./data/datasets/conll_2003/', 'eng.train', ConllLoader.TSV_DELIM,
                          ConllLoader.DEFAULT_VOCABULARY_SIZE, ConllLoader.DEFAULT_MAX_DATA_LENGTH)