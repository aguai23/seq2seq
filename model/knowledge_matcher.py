import tensorflow as tf
from knowledge_tree import KnowledgeTree
import json
import numpy as np


class KnowledgeMatcher(object):

  def __init__(self, knowledge_tree, vocab, word_embedding=100, max_answer=5, max_sequence=30,
               hidden_size=256, topic_size=256, max_entity=5):

    # knowledge tree object
    self.knowledge_tree = knowledge_tree

    # dimension of word embedding
    self.word_embedding = word_embedding

    # vocabulary object
    self.vocab = vocab

    # hidden size of bi-lstm
    self.hidden_size = hidden_size

    # size of topic vector
    self.topic_size = topic_size

    # max answer sequences to match
    self.max_answer = max_answer

    self.max_sequence = max_sequence

    # max entity numbers of one node
    self.max_entity = max_entity

    # total entity and knowledge numbers in the knowledge tree
    self.entity_number = None
    self.knowledge_number = None

    # record the tree structure, every element is a tuple indicate start and end index,
    # length == number of first level node
    self.children_index = []

    # get knowledge and entity encodings
    self.entity_encodings, self.knowledge_encodings = self.get_knowledge_embedding()

    # print(self.entity_embedding)
    # print(self.knowledge_embedding)
    # print(self.children_index)

    # the data needs to be feed
    self.question = tf.placeholder(tf.int32, shape=[None, max_sequence])
    self.answer = tf.placeholder(tf.int32, shape=[None, max_answer, max_sequence])
    self.answer_mask = tf.placeholder(tf.float32, shape=[None, max_answer])
    self.stop_label = tf.placeholder(tf.float32, shape=[None, max_answer])

    # build model
    self.question_feature = None

    # used for debug
    self.test_output = None

    self.question_attentions, answer_attentions, topic_vectors, self.stop_masks = self.knowledge_attention()

    self.output_logits, self.output_labels, self.output_masks = self.answer_generation(self.question_attentions, topic_vectors)

    self.answers = self.decoding(self.question_attentions, topic_vectors)

    self.cost = self.build_cost(self.question_attentions, answer_attentions, self.stop_masks,
                                self.output_logits, self.output_labels, self.output_masks)

    print("----------------trainable variables------------------")
    for trainable_variable in tf.trainable_variables():
      print(trainable_variable)

  def answer_generation(self, question_attentions, topic_vectors):
    """
    generate answers, for training, because we use true label here, not a decoding process
    :param question_attentions: the attention values on knowledge graph
    :param topic_vectors: topic vector of each step
    :return: generated output, and target output
    """

    with tf.variable_scope("answer_generation", reuse=tf.AUTO_REUSE):

      answer_segs = tf.split(self.answer, self.max_answer, axis=1)

      output_logits = []
      output_labels = []
      output_masks = []

      for (answer_seg, topic_vector, question_attention) in zip(answer_segs, topic_vectors, question_attentions):
        target_input = tf.squeeze(answer_seg, [1])
        target_output = tf.roll(target_input, shift=-1, axis=1)

        target_mask = target_output > self.vocab["[PAD]"]
        input_mask = tf.logical_and(target_input > self.vocab["[PAD]"], target_input != self.vocab["[SEP]"])
        target_mask = tf.logical_and(target_mask, input_mask)
        output_masks.append(tf.to_float(target_mask))

        target_input = tf.one_hot(target_input, depth=len(self.vocab))
        target_output = tf.one_hot(target_output, depth=len(self.vocab))

        feature_vector = self.build_decoding_feature(question_attention, topic_vector)

        # print(topic_vector)
        generation_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, name="generation_cell", reuse=tf.AUTO_REUSE)
        initial_state = tf.nn.rnn_cell.LSTMStateTuple(feature_vector, feature_vector)
        outputs, _ = tf.nn.dynamic_rnn(generation_cell, target_input, initial_state=initial_state)
        # print(outputs)
        logits = tf.layers.dense(outputs, len(self.vocab), activation=None, name="vocab_output", reuse=tf.AUTO_REUSE)
        output_logits.append(logits)
        output_labels.append(target_output)

    return output_logits, output_labels, output_masks

  def build_decoding_feature(self, question_attention, topic_vector):
    """
    build feature vector for decoder, concat input feature, knowledge attention and topic vector
    :param question_attention: question attention value
    :param topic_vector: topic vector
    :return: feature vector
    """
    # extract attended knowledge
    question_knowledge_attention = question_attention[1]
    batch_size = tf.shape(question_knowledge_attention)[0]
    tiled_knowledge = tf.tile(tf.expand_dims(self.knowledge_encodings, axis=0), [batch_size, 1, 1])
    feature_vector = tf.multiply(tiled_knowledge, tf.expand_dims(question_knowledge_attention, axis=2))
    feature_vector = tf.reduce_sum(feature_vector, axis=1)
    feature_vector = tf.concat([feature_vector, topic_vector, self.question_feature], axis=1)
    # print(feature_vector)

    feature_vector = tf.layers.dense(feature_vector, self.hidden_size, name="feature_squeeze", reuse=tf.AUTO_REUSE)
    return feature_vector

  def decoding(self, question_attentions, topic_vectors):
    """
    decoding process, used for evaluation
    :param question_attentions: question encodings
    :param topic_vectors: topic vectors
    :return: decoded output
    """

    with tf.variable_scope("answer_generation", reuse=tf.AUTO_REUSE):

      answers = []
      for (topic_vector, question_attention) in zip(topic_vectors, question_attentions):

        feature_vector = self.build_decoding_feature(question_attention, topic_vector)
        lstm_state = tf.nn.rnn_cell.LSTMStateTuple(feature_vector, feature_vector)
        generation_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, name="generation_cell", reuse=tf.AUTO_REUSE)

        # build input sequence
        batch_size = tf.shape(self.question)[0]
        input_sequence = tf.convert_to_tensor([self.vocab["[CLS]"]], dtype=tf.int32)
        input_sequence = tf.tile(tf.expand_dims(input_sequence, axis=0), [batch_size, 1])
        input_sequence = tf.one_hot(input_sequence, depth=len(self.vocab))

        initial_character = input_sequence
        for i in range(self.max_sequence):
          outputs, lstm_state = tf.nn.dynamic_rnn(generation_cell, input_sequence, initial_state=lstm_state)
          logits = tf.layers.dense(outputs, len(self.vocab), activation=tf.nn.softmax, name="vocab_output",
                                   reuse=tf.AUTO_REUSE)
          logits = tf.one_hot(tf.argmax(logits, axis=2), depth=len(self.vocab))
          input_sequence = tf.concat([initial_character, logits], axis=1)
        final_logits = tf.argmax(logits, axis=2)
        # print(final_logits)
        answers.append(final_logits)
      answers = tf.transpose(tf.convert_to_tensor(answers, dtype=tf.int32), [1, 0, 2])
      print(answers)
    return answers

  def get_word_embeddings(self):
    """
    get word embeddings, params of vocab_size * word_embedding
    :return: params
    """
    with tf.variable_scope("word_embedding", reuse=tf.AUTO_REUSE):
      vocab_size = len(self.vocab)
      word_embeddings = tf.get_variable("word_embedding", [vocab_size, self.word_embedding])
    return word_embeddings

  def get_knowledge_embedding(self):
    """
    convert knowledge graph to encoding vectors, then extract features with rnn
    :return:entity_encoding [entity_number, word_embedding], knowledge_encoding [knowledge_number, 4 * word_embedding]
    """
    word_embeddings = self.get_word_embeddings()
    entity_embeddings = []
    knowledge_embeddings = []

    # convert to word embeddings
    for entity_node in self.knowledge_tree.root.children:
      word_id = entity_node.embedding
      entity_embedding = tf.nn.embedding_lookup(word_embeddings, word_id)
      entity_embeddings.append(entity_embedding)
      start = len(knowledge_embeddings)
      for knowledge_node in entity_node.children:
        knowledge_id = knowledge_node.embedding
        knowledge_embedding = tf.nn.embedding_lookup(word_embeddings, knowledge_id)
        knowledge_embeddings.append(knowledge_embedding)
      end = len(knowledge_embeddings)
      self.children_index.append((start, end))
    self.entity_number = len(entity_embeddings)
    self.knowledge_number = len(knowledge_embeddings)
    entity_encodings = []

    # extract feature
    with tf.variable_scope("knowledge_encoding"):

      # extract entity feature with self attention
      for entity_embedding in entity_embeddings:
        entity_encoding = self.entity_attention(entity_embedding)
        entity_encodings.append(entity_encoding)

      # feed knowledge encoding to bi-lstm
      forward_knowledge_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, state_is_tuple=False,
                                                       name="forward_knowledge_cell",
                                                       reuse=tf.AUTO_REUSE)
      backward_knowledge_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, state_is_tuple=False,
                                                        name="backward_knowledge_cell",
                                                        reuse=tf.AUTO_REUSE)
      _, knowledge_feature = tf.nn.bidirectional_dynamic_rnn(forward_knowledge_cell, backward_knowledge_cell,
                                                             tf.convert_to_tensor(knowledge_embeddings,
                                                                                  dtype=tf.float32)
                                                             , dtype=tf.float32)
      knowledge_encodings = tf.concat([knowledge_feature[0], knowledge_feature[1]], axis=-1)

    entity_encodings = tf.convert_to_tensor(entity_encodings, dtype=tf.float32)
    knowledge_encodings = tf.convert_to_tensor(knowledge_encodings, dtype=tf.float32)
    return entity_encodings, knowledge_encodings

  def entity_attention(self, entity_embedding):
    """
    encoding entity list in the knowledge graph with self attention
    :param entity_embedding: list of entity embedding, [max_entity, word_embedding]
    :return: [1, word_embedding]
    """
    embeddings = tf.split(entity_embedding, self.max_entity, axis=0)
    attention_vector = None
    for embedding in embeddings:
      # print(embedding)
      attention_value = tf.layers.dense(embedding, 1, reuse=tf.AUTO_REUSE, name="entity_attention",
                                        activation=tf.nn.relu)
      if attention_vector is None:
        attention_vector = attention_value
      else:
        attention_vector = tf.concat([attention_vector, attention_value], axis=0)
    attention_vector = tf.nn.softmax(attention_vector)
    # print("attention vector" + str(attention_vector))
    attention_entity = tf.multiply(entity_embedding, attention_vector)
    # print(attention_entity)
    attention_entity = tf.reduce_sum(attention_entity, axis=0)
    return attention_entity

  def knowledge_attention(self):
    """
    make question and each answer part attend on each entity and knowledge
    :return: question attention(entity + knowledge), answer attention, topic vectors, stop mask
    """
    with tf.variable_scope("knowledge_attention"):
      # get word embedding for questions and answers
      word_embeddings = self.get_word_embeddings()
      question_embedding = tf.nn.embedding_lookup(word_embeddings, self.question)
      answer_embedding = tf.nn.embedding_lookup(word_embeddings, self.answer)
      with tf.variable_scope("sentence_embedding"):
        # calculate question encoding with Bi-LSTM
        forward_question_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, state_is_tuple=False,
                                                        name="forward_question_cell",
                                                        reuse=tf.AUTO_REUSE)
        backward_question_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, state_is_tuple=False,
                                                         name="backward_question_cell",
                                                         reuse=tf.AUTO_REUSE)
        _, question_feature = tf.nn.bidirectional_dynamic_rnn(forward_question_cell, backward_question_cell,
                                                              question_embedding, dtype=tf.float32)
        question_feature = tf.concat([question_feature[0], question_feature[1]], axis=-1)

        self.question_feature = question_feature
        # print(question_feature)

        with tf.variable_scope("sequential_answer"):
          # initialize topic vectors of zeros
          topic_vector = np.zeros((1, self.topic_size), dtype=np.float32)
          batch_size = tf.shape(question_feature)[0]
          topic_vector = tf.tile(topic_vector, [batch_size, 1])
          topic_vectors = []

          question_attentions = []
          answer_attentions = []
          stop_masks = []
          # start answer generation steps
          for i in range(self.max_answer):
            answer_seg = answer_embedding[:, i, :, :]

            forward_answer_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, state_is_tuple=False,
                                                          name="forward_answer_cell",
                                                          reuse=tf.AUTO_REUSE)
            backward_answer_cell = tf.nn.rnn_cell.LSTMCell(self.hidden_size, state_is_tuple=False,
                                                           name="backward_answer_cell",
                                                           reuse=tf.AUTO_REUSE)
            _, answer_feature = tf.nn.bidirectional_dynamic_rnn(forward_answer_cell, backward_answer_cell,
                                                                answer_seg, dtype=tf.float32)
            answer_feature = tf.concat([answer_feature[0], answer_feature[1]], axis=1)

            # get topic vector
            topic_vector = self.generate_topic(question_feature, topic_vector)
            topic_vectors.append(topic_vector)

            # get stop sign
            stop_mask = self.stop_mask(question_feature, topic_vector)
            stop_masks.append(stop_mask)

            question_entity_attention, question_knowledge_attention = \
              self.knowledge_match(tf.concat([question_feature, topic_vector], axis=1), "question_match", True)

            answer_entity_attention, answer_knowledge_attention = \
              self.knowledge_match(answer_feature, "answer_match", False)

            question_attentions.append((question_entity_attention, question_knowledge_attention))
            answer_attentions.append((answer_entity_attention, answer_knowledge_attention))

          stop_masks = tf.convert_to_tensor(stop_masks, dtype=tf.float32)
          stop_masks = tf.squeeze(stop_masks, [2])
          stop_masks = tf.transpose(stop_masks, [1, 0])
        return question_attentions, answer_attentions, topic_vectors, stop_masks

  def stop_mask(self, sentence_feature, prev_topic):

    with tf.variable_scope("stop_mask", reuse=True):
      stop_sign = tf.layers.dense(tf.concat([sentence_feature, prev_topic], axis=1), 1,
                                  activation=None, reuse=tf.AUTO_REUSE)
      return stop_sign

  def generate_topic(self, sentence_feature, prev_topic):
    """
    generate topic vector
    :param sentence_feature: question feature
    :param prev_topic: previous topic vector
    :return: next topic vector
    """

    with tf.variable_scope("topic_generator", reuse=True):
      topic_vector = tf.layers.dense(tf.concat([sentence_feature, prev_topic], axis=1), self.topic_size,
                                     activation=tf.nn.tanh, reuse=tf.AUTO_REUSE)
      return topic_vector

  def knowledge_match(self, sentence_feature, scope, is_question):
    """
    given a question sentence or an answer sentence, return the attention value of each entity node
    and knowledge node
    :param sentence_feature:[batch_size, 4 * hidden size], be aware of LSTM and bi directional
    :param scope: answer or question
    :param is_question:
    :return: entity attention [batch_size, entity_number], knowledge attention [batch_size, knowledge_number]
    """
    with tf.variable_scope(scope):
      sentence_feature = tf.layers.dense(sentence_feature, self.hidden_size, name="sentence_encoding",
                                         reuse=tf.AUTO_REUSE)
      entity_feature = tf.layers.dense(self.entity_encodings,
                                       self.hidden_size, name="entity_encoding",
                                       reuse=tf.AUTO_REUSE)
      entity_features = tf.split(entity_feature, self.entity_number, axis=0)
      entity_attention = None
      batch_size = tf.shape(sentence_feature)[0]
      for entity_encoding in entity_features:
        entity_encoding = tf.tile(entity_encoding, [batch_size, 1])
        attention_value = tf.layers.dense(tf.multiply(entity_encoding, sentence_feature), 1, name="entity_match",
                                          reuse=tf.AUTO_REUSE)
        if entity_attention is None:
          entity_attention = attention_value
        else:
          entity_attention = tf.concat([entity_attention, attention_value], axis=1)
      # print(entity_attention)
      # print(sentence_feature)
      # print(entity_feature)
      knowledge_feature = tf.layers.dense(self.knowledge_encodings,
                                          self.hidden_size,
                                          name="knowledge_encoding",
                                          reuse=tf.AUTO_REUSE)
      knowledge_features = tf.split(knowledge_feature, self.knowledge_number, axis=0)
      knowledge_attention = None
      for knowledge_encoding in knowledge_features:
        knowledge_encoding = tf.tile(knowledge_encoding, [batch_size, 1])
        attention_value = tf.layers.dense(tf.multiply(knowledge_encoding, sentence_feature), 1, name="knowledge_match",
                                          reuse=tf.AUTO_REUSE)
        if knowledge_attention is None:
          knowledge_attention = attention_value
        else:
          knowledge_attention = tf.concat([knowledge_attention, attention_value], axis=1)

      # print(knowledge_attention)

      # implement multi level attention
      if is_question:
        top_down_attention = []
        for i in range(self.entity_number):
          start = self.children_index[i][0]
          end = self.children_index[i][1]
          for j in range(end - start):
            top_down_attention.append(entity_attention[:, i])
        top_down_attention = tf.transpose(tf.convert_to_tensor(top_down_attention, dtype=tf.float32), [1, 0])
        knowledge_attention = tf.multiply(knowledge_attention, top_down_attention)
        # print(knowledge_attention)
      else:
        bottom_up_attention = []
        for i in range(self.entity_number):
          start = self.children_index[i][0]
          end = self.children_index[i][1]
          bottom_up_attention.append(tf.reduce_mean(knowledge_attention[:, start:end], axis=1))
        bottom_up_attention = tf.convert_to_tensor(bottom_up_attention, dtype=tf.float32)
        bottom_up_attention = tf.transpose(bottom_up_attention, [1, 0])
        # print(bottom_up_attention)
        entity_attention = tf.multiply(bottom_up_attention, entity_attention)
        # print(entity_attention)
    return entity_attention, knowledge_attention

  def build_cost(self, question_attentions, answer_attentions, stop_masks, output_logits, output_labels, output_masks):
    """
    loss function of the whole model, consisting of several parts
    :param question_attentions: attention value on knowledge embedding of questions
    :param answer_attentions: attention value of answers
    :param stop_masks: predicted stop masks
    :param output_logits: output logit
    :param output_labels: output label
    :param output_masks: mask
    :return: loss value
    """
    # build matching loss
    match_loss = []
    for i in range(self.max_answer):
      question_entity_attention = question_attentions[i][0]
      question_knowledge_attention = answer_attentions[i][1]
      answer_entity_attention = answer_attentions[i][0]
      answer_knowledge_attention = answer_attentions[i][1]

      entity_loss = tf.reduce_sum(tf.square(tf.subtract(question_entity_attention, answer_entity_attention)), axis=1)

      knowledge_loss = tf.reduce_sum(tf.square(tf.subtract(question_knowledge_attention, answer_knowledge_attention)),
                                     axis=1)

      matching_loss = tf.add(entity_loss, knowledge_loss)
      match_loss.append(matching_loss)
    match_loss = tf.transpose(tf.convert_to_tensor(match_loss, dtype=tf.float32), [1, 0])
    # print(match_loss)
    match_loss = tf.multiply(match_loss, self.answer_mask)
    match_loss = tf.reduce_mean(match_loss)

    # build stop loss
    # print(stop_masks)
    stop_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=stop_masks, labels=self.stop_label)
    stop_loss = tf.multiply(stop_loss, self.answer_mask)
    stop_loss = tf.reduce_mean(stop_loss)

    # build generation loss
    # print(output_labels)
    # print(output_logits)
    generation_loss = []
    for (output_logit, output_label, output_mask) in zip(output_logits, output_labels, output_masks):
      sentence_loss = tf.nn.softmax_cross_entropy_with_logits(logits=output_logit, labels=output_label)
      sentence_loss = tf.multiply(sentence_loss, output_mask)
      generation_loss.append(tf.reduce_mean(sentence_loss, axis=1))

    generation_loss = tf.transpose(tf.convert_to_tensor(generation_loss, dtype=tf.float32), [1,0])
    # print(generation_loss)
    generation_loss = tf.multiply(generation_loss, self.answer_mask)
    generation_loss = tf.reduce_mean(generation_loss)

    loss = match_loss + stop_loss + generation_loss
    return loss


# used for test
if __name__ == "__main__":
  knowledge_tree = KnowledgeTree("./data/diabetes_knowledge.txt", "糖尿病", "./data/dict.txt")
  vocab = json.load(open("./data/vocab.json"))
  knowledge_tree.extract_embedding(vocab)
  knowledge_matcher = KnowledgeMatcher(knowledge_tree, vocab)
