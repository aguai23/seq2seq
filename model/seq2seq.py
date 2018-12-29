import tensorflow as tf
import numpy as np
from data_processor import DataProcessor


class Seq2Seq(object):

  def __init__(self, start_token, vocab_embedding, embedding_size=100, hidden_size=256, max_question_token=30, max_answer_token=100):

    self.embedding_size = embedding_size

    self.hidden_size = hidden_size

    self.vocab_size = vocab_embedding.shape[0]

    self.max_question_token = max_question_token

    self.max_answer_token = max_answer_token

    self.vocab_embedding = tf.get_variable("vocab_embedding", [self.vocab_size, self.embedding_size], dtype=tf.float32)

    self.start_token = tf.convert_to_tensor(start_token, dtype=tf.int32)

    self.question = tf.placeholder(shape=[None, self.max_question_token], dtype=tf.int32)
    self.answer = tf.placeholder(shape=[None, self.max_answer_token], dtype=tf.int32)
    self.answer_mask = tf.placeholder(shape=[None, self.max_answer_token], dtype=tf.float32)
    self.answer_label = tf.placeholder(tf.int32, shape=[None, self.max_answer_token])

    context = self.encode(self.question)
    self.outputs = self.decode_train(self.answer, context)
    self.output_tokens = tf.argmax(tf.nn.softmax(self.outputs, axis=-1), axis=-1)
    self.infer_outputs = self.decode_infer(context)
    print(self.infer_outputs)
    # print(outputs)

    self.cost = self.build_cost(self.outputs)

    print("----------------trainable variables------------------")
    for trainable_variable in tf.trainable_variables():
      print(trainable_variable)

  def encode(self, question):
    """
    encoding part
    :param question: [batch_size, max_question_token, word_embedding]
    :return: context tuple
    """

    with tf.variable_scope("encoder"):
      question_embedding = tf.nn.embedding_lookup(self.vocab_embedding, question)
      encode_forward_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, name="encoder_forward_cell", reuse=tf.AUTO_REUSE)
      encoder_backward_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, name="encoder_backward_cell", reuse=tf.AUTO_REUSE)

      _, context = tf.nn.bidirectional_dynamic_rnn(encode_forward_cell, encoder_backward_cell, question_embedding, dtype=tf.float32)

      context = tf.add(context[0], context[1])

    return context

  def decode_train(self, answer, context):

    with tf.variable_scope("decoder"):

      answer_embedding = tf.nn.embedding_lookup(self.vocab_embedding, answer)
      decoder_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, name="decoder_cell", reuse=tf.AUTO_REUSE)
      state = tf.nn.rnn_cell.LSTMStateTuple(context[0], context[1])
      answer_tokens = tf.split(answer_embedding, self.max_answer_token, axis=1)

      outputs = []
      for i in range(self.max_answer_token):
        input_token = tf.squeeze(answer_tokens[i], axis=1)
        output, state = decoder_cell(input_token, state)
        output = tf.layers.dense(output, self.vocab_size, activation=None, reuse=tf.AUTO_REUSE, name="to_vector")
        outputs.append(output)

      outputs = tf.convert_to_tensor(outputs, dtype=tf.float32)
      outputs = tf.transpose(outputs, [1, 0, 2])

    return outputs

  def decode_infer(self, context):

    with tf.variable_scope("decoder"):
      decoder_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, name="decoder_cell", reuse=tf.AUTO_REUSE)
      state = tf.nn.rnn_cell.LSTMStateTuple(context[0], context[1])

      batch_size = tf.shape(self.answer)[0]
      print(self.start_token)
      initial_input = tf.tile(tf.expand_dims(self.start_token, axis=0), [batch_size])
      # initial_input = tf.expand_dims(initial_input, axis=1)
      # print(initial_input)
      input_seq = tf.nn.embedding_lookup(self.vocab_embedding, initial_input)

      outputs = []

      for i in range(self.max_answer_token):
        # print(input_seq)

        output, state = decoder_cell(input_seq, state)
        output = tf.layers.dense(output, self.vocab_size, activation=None, reuse=tf.AUTO_REUSE, name="to_vector")
        output = tf.argmax(tf.nn.softmax(output, axis=-1), axis=-1)
        # append to output
        outputs.append(output)
        input_seq = tf.nn.embedding_lookup(self.vocab_embedding, output)
      outputs = tf.convert_to_tensor(outputs, dtype=tf.float32)
      outputs = tf.transpose(outputs, [1, 0])
    return outputs

  def build_cost(self, outputs):
    # target_label = tf.one_hot(self.answer_label, depth=self.vocab_size, dtype=tf.float32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=outputs, labels=self.answer_label)
    loss = tf.multiply(self.answer_mask, loss)
    loss = tf.reduce_mean(loss)
    return loss


if __name__ == "__main__":
  data_processor = DataProcessor("./data/QA_data/varicocele/", "./data/QA_data/varicocele/varicocele.json",
                                 word2vec="./data/word2vec/varicocele")
  seq2seq = Seq2Seq(data_processor.start_token, data_processor.vocab_embedding)