import time

import os

import config
import dataset
from gensim.models import Word2Vec, Doc2Vec, LdaModel


def create_w2v_model(sentences):
    config.logger.info('creating w2v model ...')

    model = Word2Vec(size=config.embed_size, window=5, min_count=1, workers=4)
    model.build_vocab(sentences)

    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    model.save(config.w2v_model_file)

    return model


def load_or_create_w2v_model():
    start_time = time.time()
    if config.w2v_model_file != None and os.path.exists(config.w2v_model_file):
        config.logger.info('loading w2v model from {} ...'.format(config.w2v_model_file))
        model = Word2Vec.load(config.w2v_model_file)
    else:
        sentences = dataset.load_unlabeled_data_sentences()
        model = create_w2v_model(sentences)

    config.logger.info('creating w2v model took {}'.format((time.time() - start_time)))
    return model
