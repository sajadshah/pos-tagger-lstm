import argparse
import dataset, word2vec as w2v
import config
import model

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--w2v-train',
    #     choices=[True, False],
    #     type=bool,
    #     default=False,
    #     help='whether train w2v features or use saved model')
    #
    # args = parser.parse_args()

    config.setup()

    data_size, label_to_index, num_labels, max_sentence_length = dataset.get_data_info()

    w2v_model = w2v.load_or_create_w2v_model()
    lexicon, word_to_index, index_to_word = dataset.create_lexicon_from_w2v(w2v_model)

    model.build_and_train_model(w2v_model, word_to_index, label_to_index, max_sentence_length, num_labels)

