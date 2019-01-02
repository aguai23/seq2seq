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

      _, context = tf.nn.bidirectional_dynamic_rnn(encode_forward_cell, encoder_backward_cell,
                                                   question_embedding, dtype=tf.float32)
      context_c = tf.add(context[0].c, context[1].c)
      context_h = tf.add(context[0].h, context[1].h)
      context = tf.nn.rnn_cell.LSTMStateTuple(context_c, context_h)
    return context

  def decode_train(self, answer, context):

    with tf.variable_scope("decoder"):
      answer_embedding = tf.nn.embedding_lookup(self.vocab_embedding, answer)
      decoder_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, name="decoder_cell", reuse=tf.AUTO_REUSE)
      state = context
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

  def decode_infer(self, context, beam_width=3):

    with tf.variable_scope("decoder"):
      decoder_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, name="decoder_cell", reuse=tf.AUTO_REUSE)
      state = tf.nn.rnn_cell.LSTMStateTuple(context[0], context[1])

      batch_size = tf.shape(self.answer)[0]
      initial_input = tf.tile(tf.expand_dims(self.start_token, axis=0), [batch_size])

      outputs = []
      beam_outputs = []

      for i in range(beam_width):
        beam_outputs.append((initial_input,
                             state,
                             tf.tile(tf.constant([1.0], shape=[1], dtype=tf.float32), [batch_size]),
                             None))

      for i in range(self.max_answer_token):
        # print(input_seq)
        beam_scores = None
        beam_indices = None
        beam_states_c = None
        beam_states_h = None
        beam_pre_outputs = None

        for beam_output in beam_outputs:
          inputs = beam_output[0]
          state = beam_output[1]
          score = beam_output[2]
          previous_output = beam_output[3]
          inputs = tf.nn.embedding_lookup(self.vocab_embedding, inputs)
          output, state = decoder_cell(inputs, state)
          output = tf.layers.dense(output, self.vocab_size, activation=None, reuse=tf.AUTO_REUSE, name="to_vector")
          output = tf.nn.softmax(output, axis=-1)
          values, indices = tf.math.top_k(output, k=beam_width)
          values = tf.multiply(values, tf.expand_dims(score, axis=1))

          # concat score
          if beam_scores is None:
            beam_scores = values
          else:
            beam_scores = tf.concat([beam_scores, values], axis=1)

          # concat index
          if beam_indices is None:
            beam_indices = indices
          else:
            beam_indices = tf.concat([beam_indices, indices], axis=1)

          # concat state
          if beam_states_c is None:
            beam_states_c = tf.tile(tf.expand_dims(state.c, axis=1), [1, batch_size, 1])
          else:
            beam_states_c = tf.concat([beam_states_c,
                                       tf.tile(tf.expand_dims(state.c, axis=1), [1, batch_size, 1])], axis=1)

          if beam_states_h is None:
            beam_states_h = tf.tile(tf.expand_dims(state.h, axis=1), [1, batch_size, 1])
          else:
            beam_states_h = tf.concat([beam_states_h,
                                       tf.tile(tf.expand_dims(state.h, axis=1), [1, batch_size, 1])], axis=1)

          if previous_output is not None:
            if beam_pre_outputs is None:
              beam_pre_outputs = tf.tile(tf.expand_dims(previous_output, axis=1), [1, beam_width, 1])
            else:
              beam_pre_outputs = tf.concat([beam_pre_outputs,
                                            tf.tile(tf.expand_dims(previous_output, axis=1), [1, beam_width, 1])],
                                           axis=1)

        values, indices = tf.math.top_k(beam_scores, k=beam_width)
        word_indices = tf.batch_gather(beam_indices, indices)
        beam_states_c = tf.batch_gather(beam_states_c, indices)
        beam_states_h = tf.batch_gather(beam_states_h, indices)
        # print(indices)
        # print(beam_pre_outputs)
        if beam_pre_outputs is not None:
          beam_pre_outputs = tf.batch_gather(beam_pre_outputs, indices)
          beam_pre_outputs = tf.squeeze(tf.split(beam_pre_outputs, beam_width, axis=1), axis=2)
        # print(values)
        # print(word_indices)
        # print(beam_states_c)
        # print(beam_states_h)

        beam_outputs = []
        scores = tf.split(values, beam_width, axis=1)
        outputs = tf.split(word_indices, beam_width, axis=1)
        beam_states_c = tf.split(beam_states_c, beam_width, axis=1)
        beam_states_h = tf.split(beam_states_h, beam_width, axis=1)

        # print(outputs)
        # print(beam_states_c)
        # print(scores)
        # print(outputs[0])
        for i in range(beam_width):
          score = tf.squeeze(scores[i], axis=1)
          state_c = tf.squeeze(beam_states_c[i], axis=1)
          state_h = tf.squeeze(beam_states_h[i], axis=1)
          if beam_pre_outputs is not None:
            beam_outputs.append((tf.squeeze(outputs[i], axis=1),
                                tf.nn.rnn_cell.LSTMStateTuple(state_c, state_h),
                                score,
                                tf.concat([beam_pre_outputs[i], outputs[i]], axis=1)))
          else:
            beam_outputs.append((tf.squeeze(outputs[i], axis=1),
                                 tf.nn.rnn_cell.LSTMStateTuple(state_c, state_h),
                                 score,
                                 outputs[i]))
        # print(beam_outputs[0][3])
        # append to output
      #   outputs.append(output)
      #   input_seq = tf.nn.embedding_lookup(self.vocab_embedding, output)
      # outputs = tf.convert_to_tensor(outputs, dtype=tf.float32)
      # outputs = tf.transpose(outputs, [1, 0])
    return beam_outputs[0][3]

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