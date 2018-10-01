import os, io
import nltk
import random
import numpy as np
import pickle
import config


def load_data():
    if config.saved_data_file != None and os.path.exists(config.saved_data_file):
        return np.load(config.saved_data_file)

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
    np.save(config.saved_data_file, data)
    return data


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
    data = load_data()
    lexicon = []
    for w, l in data:
        if w not in lexicon:
            lexicon.append(w)
    lexicon = np.asarray(lexicon)

    np.save(config.lexicon_file, lexicon)
    return lexicon



def load_or_create_lexicon():
    if config.lexicon_file != None and os.path.exists(config.lexicon_file):
        return np.load(config.lexicon_file)

    return create_lexicon()






