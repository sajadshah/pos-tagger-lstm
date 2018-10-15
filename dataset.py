import os, io
import time
import numpy as np
import config
import nltk
import random
import pickle

delimiter_labels = ['DELIM', 'FILE_DELIM']


def load_data():
    config.logger.info('loading data ...')
    start_time = time.time()
    if config.saved_data_file != None and os.path.exists(config.saved_data_file):
        data = np.load(config.saved_data_file)
        config.logger.info('loading data took {} (from saved, {} words)'.format((time.time() - start_time), 1, len(data)))
        return data

    lbl_files = os.listdir(config.lbl_data_dir)
    data = []
    for lbl_file in lbl_files:
        f = io.open(os.path.join(config.lbl_data_dir, lbl_file), mode='r', encoding='utf-8')
        lines = f.readlines()
        for i in range(1,len(lines)):
            tokens = lines[i].split()
            label = tokens[-1]
            if label == 'SUBJ':
                continue
            word = ' '.join(tokens[:-1])
            data.append((word, label))
        data.append((';', 'FILE_DELIM'))
        f.close()

    data = np.asarray(data)
    config.logger.info('loading data took {} ({} files, {} words)'.format((time.time() - start_time), len(lbl_files), len(data)))
    np.save(config.saved_data_file, data)

    return data


def is_delim(word, label):
    return word == '.' or word == '#' or label == 'FILE_DELIM'


def load_unlabeled_data_sentences():
    config.logger.info('loading unlabeled data sentences ...')
    start_time = time.time()

    data = load_data()
    sentences = []
    sentence = []
    for w, l in data:
        sentence.append(w)
        if is_delim(w, l):
            if len(sentence) > 1:
                sentences.append(sentence)
            sentence = []

    config.logger.info('loading unlabeled data sentences took {} ({} sentences)'.format((time.time() - start_time), len(sentences)))
    return sentences

def load_labeled_data_sentences():
    pass

def load_iter(lbl_data_dir, file_batch_count = 50):
    lbl_files = os.listdir(lbl_data_dir)
    data = []
    for id, lbl_file in enumerate(lbl_files):
        f = io.open(os.path.join(lbl_data_dir, lbl_file), mode='r', encoding='utf-8')
        lines = f.readlines()
        for i in range(1, len(lines)):
            tokens = lines[i].split()
            label = tokens[-1]
            word = ' '.join(tokens[:-1])
            data.append((word, label))
        f.close()
        if id % file_batch_count == 0 or id == len(lbl_files) - 1:
            print('Next {} files from {} loaded {} sentences'.format(id, len(lbl_files), len(data)))
            yield data
            data = []


def create_lexicon():
    config.logger.info('creating lexicon ...')
    start_time = time.time()
    data = load_data()
    lexicon = []
    for w, l in data:
        if w not in lexicon:
            lexicon.append(w)
    lexicon = np.asarray(lexicon)

    np.save(config.lexicon_file, lexicon)
    config.logger.info('creating lexicon took {} ({} words)'.format((time.time() - start_time), len(lexicon)))
    return lexicon



def load_or_create_lexicon():
    config.logger.info('loading lexicon ...')
    start_time = time.time()
    if config.lexicon_file != None and os.path.exists(config.lexicon_file):
        lexicon = np.load(config.lexicon_file)
    else:
        lexicon = create_lexicon()

    config.logger.info('loading lexicon took {} ({} words)'.format((time.time() - start_time), len(lexicon)))
    return lexicon






