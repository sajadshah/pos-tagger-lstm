import tensorflow as tf
import config

def build_model(input):
    pass


def model_inputs():
    inputs_ = tf.placeholder(tf.int32, [None, None], name='inputs')
    labels_ = tf.placeholder(tf.int32, [None, None], name='labels')

    return inputs_, labels_


def build_lstm_layers(lstm_sizes, inputs_, batch_size):
    lstms = [tf.nn.rnn_cell.LSTMCell(size) for size in lstm_sizes]
    # Add dropout to the cell
    # drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob_) for lstm in lstms]

    # Stack up multiple LSTM layers, for deep learning
    cell = tf.nn.rnn_cell.MultiRNNCell(lstms)
    # Getting an initial state of all zeros
    initial_state = cell.zero_state(batch_size, tf.float32)

    lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, inputs_, initial_state=initial_state)

    return initial_state, lstm_outputs, cell, final_state


def build_cost_fn_and_opt(lstm_outputs, labels_, learning_rate):
    predictions = tf.contrib.layers.fully_connected(lstm_outputs[:, -1], 1, activation_fn=tf.sigmoid)
    loss = tf.losses.mean_squared_error(labels_, predictions)
    optimzer = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss)

    return predictions, loss, optimzer


def build_accuracy(predictions, labels_):
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return accuracy


def build_and_train_model(lexicon, w2v_model, num_labels):
    inputs_, labels_ = model_inputs()

    initial_state, lstm_outputs, lstm_cell, final_state = build_lstm_layers(config.lstm_sizes, inputs_, config.batch_size)
    predictions, loss, optimizer = build_cost_fn_and_opt(lstm_outputs, labels_, learning_rate)
    accuracy = build_accuracy(predictions, labels_)

