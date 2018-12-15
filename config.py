import os
import logging

#DATA_DIR_NAME = 'BijanKhan_Full_Corpus'
DATA_DIR_NAME = 'bijankhan_lbl'

data_dir = os.path.join(os.path.curdir, 'data')
lbl_data_dir = os.path.join(data_dir, DATA_DIR_NAME)

exp_dir = os.path.join(os.path.curdir, 'exp', DATA_DIR_NAME)

log_file = os.path.join(exp_dir, 'log.txt')
save_dir = os.path.join(exp_dir, 'save')

lexicon_file = os.path.join(exp_dir, 'lexicon.npy')
saved_data_file = os.path.join(exp_dir, 'data.npy')
encoded_data = os.path.join(exp_dir, 'encoded_data.npy')
w2v_model_file = os.path.join(exp_dir, 'word2vec.model')

embed_size = 100
batch_size = 128
lstm_sizes = [512, 512]
learning_rate = 0.001

logger = None

def make_dirs():
    for x in [exp_dir, save_dir]:
        if not os.path.isdir(x):
            os.makedirs(x)

def log_setup():
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create a file handler
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)

    # create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(handler)

    logger.info('Hello baby!')


def setup():
    make_dirs()
    log_setup()


def get_last_model_file():
    return os.path.join(save_dir, "last_model.ckpt")


def get_labels_file():
    return os.path.join(exp_dir, "labels.json")


def get_test_results_file():
    return os.path.join(exp_dir, "test_results.txt")