import argparse
import dataset, word2vec as w2v
import config


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

    w2v_model = w2v.load_or_create_w2v_model()

    data = dataset.load_data()


