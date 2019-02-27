import os
import tensorflow as tf
import numpy as np
import time

import config
import dataset


def build_model(input):
    pass


def model_inputs():
    inputs_ = tf.placeholder(tf.float64, [config.batch_size, None, config.embed_size], name='inputs')
    seq_length_ = tf.placeholder(tf.float64, [config.batch_size], name='seq_length')
    labels_ = tf.placeholder(tf.int64, [config.batch_size, None], name='labels')

    # inputs [batch_size, num_steps, embed_size]
    # outputs [batch_size, num_steps]

    return inputs_, seq_length_, labels_


def build_lstm_layers(lstm_sizes, inputs_, batch_size):
    lstms = [tf.nn.rnn_cell.LSTMCell(num_units) for num_units in config.lstm_sizes]
    # Add dropout to the cell
    # drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob_) for lstm in lstms]

    # Stack up multiple LSTM layers, for deep learning
    cell = tf.nn.rnn_cell.MultiRNNCell(lstms)
    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float64)

    lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, inputs_, initial_state=initial_state)

    return initial_state, lstm_outputs, cell, final_state


def build_cost_fn_and_opt(lstm_outputs, labels_, num_labels):
    logits = tf.layers.dense(lstm_outputs, num_labels)
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_, logits=logits))
    learning_rate = tf.placeholder(tf.float64,  name='learning_rate')

    optimzer = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)

    predictions = tf.nn.softmax(logits)

    return predictions, loss, optimzer, learning_rate


# def build_accuracy(predictions, labels_, seq_length_):
#     prediction_index = tf.argmax(predictions, axis=2)
#     correct_pred = tf.equal(prediction_index, labels_)
#     correct_pred_count = 0
#
#     for i in range(labels_.shape[0]):
#
#     accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
#     return accuracy


def calculate_accuracy(prediction_probs, labels, sequence_length):
    prediction_index = np.argmax(prediction_probs, axis=2)
    total_count = np.sum(sequence_length)
    correct_count = 0
    print(prediction_index.shape, labels.shape)
    assert prediction_index.shape == labels.shape

    for i in range(prediction_index.shape[0]):
        seq_label = labels[i][-sequence_length[i]:]
        seq_pred = prediction_index[i][-sequence_length[i]:]
        correct_seq = np.equal(seq_label, seq_pred)
        correct_count += np.sum(np.asarray(correct_seq, dtype=np.int))

    return 1.0 * correct_count / total_count


def build_and_train_model(w2v_model, word_to_index, label_to_index, max_sentence_length, num_labels):
    inputs_, seq_length_, labels_ = model_inputs()

    initial_state, lstm_outputs, lstm_cell, final_state = build_lstm_layers(config.lstm_sizes, inputs_, config.batch_size)
    predictions, loss, optimizer, learning_rate = build_cost_fn_and_opt(lstm_outputs, labels_, num_labels)
    # accuracy = build_accuracy(predictions, labels_, seq_length_)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    last_model_file = config.get_last_model_file()
    if os.path.exists(last_model_file):
        saver.restore(sess, last_model_file)

    iter = 0
    start_time_total = time.time()
    learning_rate_value = config.start_learning_rate
    for sentences, labels in dataset.load_labeled_data_sentences_batch(config.batch_size, shuffle_files=True):
        start_time = time.time()
        features, sequence_length, labels = dataset.encoded_sentences_labels(sentences, labels, w2v_model, word_to_index, label_to_index,
                                                        max_sentence_length)
        load_batch_time = time.time() - start_time

        start_time = time.time()
        _, loss_train, prediction_probs = sess.run([optimizer, loss, predictions], feed_dict={inputs_: features, labels_: labels, seq_length_:sequence_length, learning_rate: learning_rate_value})
        learning_time = time.time() - start_time

        start_time = time.time()
        accuracy_train = calculate_accuracy(prediction_probs, labels, sequence_length)
        accuracy_calc_time = time.time() - start_time

        start_time = time.time()
        save_path = saver.save(sess, config.get_last_model_file())
        saving_time = time.time() - start_time

        config.logger.info("loss, accuracy at iteration {:0>4}: {:8.4f} , {:.2f}".format(iter, loss_train, accuracy_train*100))
        print("load time: {}, learn time: {}, save time: {}, acc time: {}, total time: {}".format(load_batch_time, learning_time, saving_time, accuracy_calc_time, (time.time() - start_time_total)))

        iter += 1
        if iter % config.every_n_iteration:
            learning_rate_value *= config.decay

def load_and_random_test_model(w2v_model, word_to_index, label_to_index, max_sentence_length, num_labels, sentence_limit = 10):
    inputs_, seq_length_, labels_ = model_inputs()

    initial_state, lstm_outputs, lstm_cell, final_state = build_lstm_layers(config.lstm_sizes, inputs_, config.batch_size)
    predictions, loss, optimizer = build_cost_fn_and_opt(lstm_outputs, labels_, num_labels, config.learning_rate)
    # accuracy = build_accuracy(predictions, labels_, seq_length_)

    prediction_index = tf.argmax(predictions, axis=2)
    correct_pred = tf.equal(prediction_index, labels_)

    sess = tf.Session()

    saver = tf.train.Saver()
    saver.restore(sess, config.get_last_model_file())

    for sentences, labels in dataset.load_labeled_data_sentences_batch(config.batch_size, shuffle_files=False):

        features, sequence_length, labels = dataset.encoded_sentences_labels(sentences, labels, w2v_model, word_to_index, label_to_index,
                                                                             max_sentence_length)

        loss_val, predictions_val, prediction_index_val, correct_pred_val = sess.run([loss, predictions, prediction_index, correct_pred], feed_dict={inputs_: features, labels_: labels, seq_length_:sequence_length})
        accuracy_val = calculate_accuracy(predictions_val, labels, sequence_length)
        config.logger.info("loss, accuracy at iteration in test time: {:8.4f} , {:.2f}".format(loss_val,
                                                                                               accuracy_val * 100))
        import codecs

        file = codecs.open(config.get_test_results_file(), "w", "utf-8")

        for i, sentence in enumerate(sentences):
            for j, w in enumerate(sentence):
                true_label = labels[i][-sequence_length[i] + j]
                predicted_label = prediction_index_val[i][-sequence_length[i] + j]
                true_label_str = dataset.label_name_from_index(true_label)
                predicted_label_str = dataset.label_name_from_index(predicted_label)
                file.write(u"{}\n".format(w))
                file.write(u"({}) [{}]\n".format(true_label, predicted_label))
                file.write(u"({}) [{}]\n\n".format(true_label_str, predicted_label_str))

            if i >= sentence_limit:
                return

        file.close()


