import tensorflow as tf
import numpy as np
from data_processor import DataProcessor
class Seq2Seq(object):

  def __init__(self, start_token, embedding_size=100, hidden_size=256, max_question_token=30, max_answer_token=100):

    self.embedding_size = embedding_size

    self.hidden_size = hidden_size

    self.max_question_token = max_question_token

    self.max_answer_token = max_answer_token

    self.start_token = tf.convert_to_tensor(start_token, dtype=tf.float32)

    self.question = tf.placeholder(shape=[None, self.max_question_token, self.embedding_size], dtype=tf.float32)
    self.answer = tf.placeholder(shape=[None, self.max_answer_token, self.embedding_size], dtype=tf.float32)
    self.answer_mask = tf.placeholder(shape=[None, self.max_answer_token], dtype=tf.float32)

    context = self.encode(self.question)
    self.outputs = self.decode_train(self.answer, context)
    self.infer_outputs = self.decode_infer(context)
    # print(self.infer_outputs)
    # print(outputs)

    self.cost = self.build_cost(self.outputs)

  def encode(self, question):

    with tf.variable_scope("encoder"):

      encode_forward_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, name="encoder_forward_cell", reuse=tf.AUTO_REUSE)
      encoder_backward_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, name="encoder_backward_cell", reuse=tf.AUTO_REUSE)

      _, context = tf.nn.bidirectional_dynamic_rnn(encode_forward_cell, encoder_backward_cell, question, dtype=tf.float32)
      context = tf.add(context[0], context[1])
    return context

  def decode_train(self, answer, context):
    with tf.variable_scope("decoder"):

      decoder_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, name="decoder_cell", reuse=tf.AUTO_REUSE)
      initial_state = tf.nn.rnn_cell.LSTMStateTuple(context[0], context[1])
      outputs, _ = tf.nn.dynamic_rnn(decoder_cell, answer, initial_state=initial_state)
      outputs = tf.layers.dense(outputs, self.embedding_size, activation=None, reuse=tf.AUTO_REUSE, name="to_vector")

    return outputs

  def decode_infer(self, context):

    outputs = None

    with tf.variable_scope("decoder"):
      batch_size = tf.shape(self.answer)[0]
      initial_input = tf.tile(tf.expand_dims(self.start_token, axis=0), [batch_size, 1])
      initial_input = tf.expand_dims(initial_input, axis=1)
      # print(initial_input)
      input_seq = initial_input
      decoder_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, name="decoder_cell", reuse=tf.AUTO_REUSE)
      state = tf.nn.rnn_cell.LSTMStateTuple(context[0], context[1])
      for i in range(self.max_answer_token):
        outputs, state = tf.nn.dynamic_rnn(decoder_cell, input_seq, initial_state=state)
        outputs = tf.layers.dense(outputs, self.embedding_size, activation=None, reuse=tf.AUTO_REUSE, name="to_vector")

        input_seq = tf.concat([initial_input, outputs], axis=1)
    return outputs

  def build_cost(self, outputs):
    target_label = tf.roll(self.answer, shift=-1, axis=1)
    square_loss = tf.square(tf.subtract(target_label, outputs))
    square_loss = tf.reduce_sum(square_loss, axis=-1)
    square_loss = tf.multiply(square_loss, self.answer_mask)
    square_loss = tf.reduce_mean(square_loss)
    return square_loss


if __name__ == "__main__":
  data_processor = DataProcessor("./data/QA_data/varicocele/", "./data/QA_data/varicocele/varicocele.json",
                                 word2vec="./data/word2vec/varicocele")
  seq2seq = Seq2Seq(data_processor.start_token)