import json
import os, io
import time
import numpy as np
import config
import itertools
from random import shuffle

delimiter_labels = ['DELIM', 'FILE_DELIM']
cache_data = None

def load_data():
    global cache_data
    config.logger.info('loading data ...')
    start_time = time.time()
    if cache_data is not None:
        return cache_data

    if config.saved_data_file != None and os.path.exists(config.saved_data_file):
        data = np.load(config.saved_data_file)
        cache_data = data
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
    cache_data = data
    return data


def encoded_sentences_labels(sentences, labels, w2v_model, word_to_index, label_to_index, max_sentence_length):
    assert len(sentences) == len(labels)

    # config.logger.info('encoding sentences and labels...')
    start_time = time.time()

    features = []
    indexed_labels = []

    for i in range(len(sentences)):
        sentence_features = []
        sentence_indexed_labels = []

        sentence_words = sentences[i]
        sentence_labels = labels[i]
        assert len(sentence_words) == len(sentence_labels)

        if len(sentence_words) > max_sentence_length:
            max_sentence_length = len(sentence_words)

        for w in sentence_words:
            try:
                word_features = w2v_model.wv[w]
            except KeyError:
                word_features = np.zeros((1, config.embed_size), np.float)
            sentence_features.append(word_features)

        for l in sentence_labels:
            try:
                label_index = label_to_index[l]
            except KeyError:
                label_index = 0
            sentence_indexed_labels.append(label_index)

        features.append(sentence_features)
        indexed_labels.append(sentence_indexed_labels)

    # features [batch_size, num_steps, embed_size]
    # labels [batch_size, num_labels]

    features_padded = np.zeros((len(features), max_sentence_length, config.embed_size), dtype=np.float64)
    labels_padded = np.zeros((len(features), max_sentence_length), dtype=np.int64)
    sequence_length = np.zeros(len(features), dtype=np.int32)

    for i, sentence_features in enumerate(features):
        features_padded[i, -len(sentence_features):, :] = np.array(sentence_features)[:max_sentence_length]
        sequence_length[i] = len(sentence_features)
    for i, sentence_indexed_labels in enumerate(indexed_labels):
        labels_padded[i, -len(sentence_indexed_labels):] = np.array(sentence_indexed_labels)[:max_sentence_length]

    # features [batch_size, max_sentence_length, embed_size]
    # labels [batch_size, num_labels]

    return features_padded, sequence_length, labels_padded


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
    config.logger.info('loading unlabeled data sentences ...')
    start_time = time.time()

    data = load_data()
    sentences = []
    labels = []
    sentence_words = []
    sentence_labels = []
    for w, l in data:
        sentence_words.append(w)
        sentence_labels.append(l)
        if is_delim(w, l):
            if len(sentence_words) > 1:
                sentences.append(sentence_words)
                labels.append(sentence_labels)
            sentence_words = []
            sentence_labels = []

    config.logger.info(
        'loading unlabeled data sentences took {} ({} sentences)'.format((time.time() - start_time), len(sentences)))
    return sentences, labels


def load_labeled_data_sentences_batch(batch_sentences_size, shuffle_files=False):
    lbl_files = os.listdir(config.lbl_data_dir)
    if shuffle_files:
        shuffle(lbl_files)
    sentences = []
    labels = []
    sentence_words = []
    sentence_labels = []
    for lbl_file in lbl_files:
        f = io.open(os.path.join(config.lbl_data_dir, lbl_file), mode='r', encoding='utf-8')
        lines = f.readlines()
        for i in range(1, len(lines)):
            tokens = lines[i].split()
            label = tokens[-1]
            if label == 'SUBJ':
                continue
            word = ' '.join(tokens[:-1])
            sentence_words.append(word)
            sentence_labels.append(label)
            if is_delim(word, label):
                if len(sentence_words) > 1:
                    sentences.append(sentence_words)
                    labels.append(sentence_labels)
                sentence_words = []
                sentence_labels = []
                if len(sentences) == batch_sentences_size:
                    yield sentences, labels
                    sentences = []
                    labels = []
        f.close()

    # yield sentences, labels


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

def create_lexicon_from_w2v(w2v_model):
    vocab = w2v_model.wv.vocab
    lexicon = vocab.keys()
    index_to_word = {}
    word_to_index = {}
    for w in vocab:
        index = vocab[w].index
        index_to_word[index] = w
        word_to_index[w] = index

    return lexicon, word_to_index, index_to_word



def get_data_info():
    sentences, labels = load_labeled_data_sentences()
    all_labels = list(itertools.chain.from_iterable(labels))
    data_size = len(sentences)
    labels_unique = np.unique(all_labels)
    label_to_index = {}
    for i, l in enumerate(labels_unique):
        label_to_index[l] = i + 1

    import json
    with open(config.get_labels_file(), 'w') as file:
        file.write(json.dumps(label_to_index))

    max_sentence_length = -1
    for s in sentences:
        if len(s) > max_sentence_length:
            max_sentence_length = len(s)

    config.logger.info('found {} labels'.format(len(labels_unique)))
    # config.logger.info(' '.join(label_to_int[:20]))
    return data_size, label_to_index, len(labels_unique), max_sentence_length


def label_name_from_index(label_index):
    with open(config.get_labels_file()) as f:
        labels = json.load(f)
        for label, index in labels.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
            if label_index == index:
                return label



