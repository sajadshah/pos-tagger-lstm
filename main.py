import argparse
import dataset, word2vec as w2v
import config
import model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--w2v-train',
        choices=[True, False],
        type=bool,
        default=False,
        help='whether train w2v features or use saved model')

    args = parser.parse_args()

    config.setup()

    lexicon = dataset.load_or_create_lexicon()
    int_to_word, word_to_int = dataset.create_lookup_tables(lexicon)
    data_size, label_to_int, num_labels = dataset.get_data_info()

    sentences, labels = dataset.load_encoded_sentences_labels(word_to_int, label_to_int)

    w2v_model = w2v.load_or_create_w2v_model()

    # data = dataset.load_data()
    model.build_and_train_model(lexicon, w2v_model, num_labels)

