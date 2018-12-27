import tensorflow as tf
import logging
import json
from model.knowledge_matcher import KnowledgeMatcher
from data_processor import DataProcessor
from knowledge_tree import KnowledgeTree
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class Trainer(object):

  def __init__(self, network, data_provider, batch_size=16, optimizer="adam", learning_rate=5e-5,
               decay_rate=1, decay_step=1000, momentum=0.9, verify_epoch=1, output_path="./saved_model/"):
    # network class
    self.network = network

    # data processor
    self.data_provider = data_provider

    # training batch size
    self.batch_size = batch_size

    # initial learning rate
    self.learning_rate = learning_rate

    self.global_step = tf.Variable(0)

    self.verify_epoch = verify_epoch

    self.output_path = output_path

    self.vocab = self.network.vocab

    self.decode_vocab = {}
    for key, value in self.vocab.items():
      self.decode_vocab[value] = key

    if optimizer == "momentum":

      learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                      global_step=self.global_step,
                                                      decay_steps=decay_step,
                                                      decay_rate=decay_rate,
                                                      staircase=True)

      self.optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate_node,
                                                  momentum=momentum).minimize(self.network.cost,
                                                                              global_step=self.global_step)
    elif optimizer == "adam":

      learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                      global_step=self.global_step,
                                                      decay_steps=decay_step,
                                                      decay_rate=decay_rate,
                                                      staircase=True)

      self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_node).minimize(self.network.cost,
                                                                                         global_step=self.global_step)

  def train(self, train_epoch=10, save_epoch=1, display_step=10, restore=False):

    train_data = self.data_provider.get_train_data()
    question = train_data["question"]
    answer = train_data["answer"]
    answer_mask = train_data["answer_mask"]
    stop_label = train_data["stop_label"]

    train_samples = question.shape[0]
    print("total training numbers " + str(train_samples))

    step_size = int(train_samples / self.batch_size)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:

      sess.run(init)

      print("--------------start training -----------------------")
      print("total training epochs " + str(train_epoch))

      if restore:
        ckpt = tf.train.get_checkpoint_state(self.output_path)
        if ckpt and ckpt.model_checkpoint_path:
          saver = tf.train.Saver()
          saver.restore(sess, ckpt.model_checkpoint_path)
          logging.info("model restored from file " + ckpt.model_checkpoint_path)

      for epoch in range(train_epoch):

        for step in range(step_size):

          start_index = self.batch_size * step
          end_index = self.batch_size * (step + 1)

          _, loss, answers, labels = sess.run([self.optimizer,
                              self.network.cost, self.network.output_logits, self.network.output_labels],
                              feed_dict={
                               self.network.question: question[start_index: end_index],
                               self.network.answer: answer[start_index: end_index],
                               self.network.answer_mask: answer_mask[
                                                         start_index: end_index],
                               self.network.stop_label: stop_label[start_index: end_index]
                             })

          if step % display_step == 0:
            logging.info("epoch {:}, step {:}, Minibatch Loss={:.4f}".format(epoch,
                                                                             step,
                                                                             loss))
            print(np.argmax(answers[0][0], axis=-1))
            print(np.argmax(labels[0][0], axis=-1))

        if epoch % save_epoch == 0:
          saver = tf.train.Saver()
          saver.save(sess, self.output_path + "epoch_" + str(epoch))
          logging.info("model saved at epoch " + str(epoch))

        if epoch % self.verify_epoch == 0:

          valid_data = data_processor.get_valid_data()
          valid_question = valid_data["question"]
          valid_answer = valid_data["answer"]
          valid_answer_mask = valid_data["answer_mask"]
          valid_stop_label = valid_data["stop_label"]

          valid_step = int(len(valid_question) / self.batch_size)
          total_loss = 0.0
          for step in range(1):
            start_index = self.batch_size * step
            end_index = self.batch_size * (step + 1)

            loss, answers, stop_masks = sess.run([self.network.cost, self.network.answers,
                                                  self.network.stop_masks],
                                                 feed_dict={
                                                   self.network.question: valid_question[start_index: end_index],
                                                   self.network.answer: valid_answer[start_index: end_index],
                                                   self.network.answer_mask: valid_answer_mask[start_index: end_index],
                                                   self.network.stop_label: valid_stop_label[start_index: end_index]
                                                 })
            print(loss)
            total_loss += loss
            decoded_answers = self.decode_answer(answers, stop_masks)
            print(decoded_answers)
          logging.info("loss on valid data " + str(total_loss / valid_step))

  def decode_answer(self, answers, stop_masks):

    decoded_answers = []

    for (answer, stop_mask) in zip(answers, stop_masks):
      decoded_sentence = ""
      for answer_seg, sentence_mask in zip(answer, stop_mask):
        for word in answer_seg:
          decoded_sentence += self.decode_vocab[word]
          if self.decode_vocab[word] == "[SEP]":
            break
        decoded_sentence += ","
        if sentence_mask < 0.5:
          break
      decoded_answers.append(decoded_sentence)

    return decoded_answers


if __name__ == "__main__":
  knowledge_tree = KnowledgeTree("./data/diabetes_knowledge.txt", "糖尿病", "./data/dict.txt")
  vocab = json.load(open("./data/vocab.json"))
  knowledge_tree.extract_embedding(vocab)
  knowledge_matcher = KnowledgeMatcher(knowledge_tree, vocab)
  data_processor = DataProcessor("./data/diabetes.json", data_dir="./data/", vocab="./data/vocab.json")
  trainer = Trainer(knowledge_matcher, data_processor)
  trainer.train(train_epoch=10, restore=True)
