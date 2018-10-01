import argparse
import os

import pickle
import dataset, word2vec as w2v
import numpy as np
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--w2v-train',
        choices=[True, False],
        type=bool,
        default=False,
        help='whether train w2v features or use saved model')


    args = parser.parse_args()

    lexicon = dataset.load_or_create_lexicon()

    data_iter = dataset.load_iter()
    w2vModel = w2v.createW2VModel()
