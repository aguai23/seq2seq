import tensorflow as tf
import logging
import json
from model.knowledge_matcher import KnowledgeMatcher
from data_processor import DataProcessor
from model.seq2seq import Seq2Seq
from knowledge_tree import KnowledgeTree
import numpy as np
from scipy import spatial
from evaluator import Evaluator
import jieba
from sklearn.utils import shuffle

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class Trainer(object):

  def __init__(self, network, data_provider, batch_size=16, optimizer="adam", learning_rate=5e-5,
               decay_rate=0.9, decay_step=1000, momentum=0.9, verify_epoch=1, output_path="./saved_model/"):
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
    answer_label = train_data["answer_label"]

    train_samples = question.shape[0]
    print("total training numbers " + str(train_samples))

    step_size = int(train_samples / self.batch_size)

    init = tf.global_variables_initializer()

    evaluator = Evaluator()

    with tf.Session() as sess:

      sess.run(init)

      print("--------------start training -----------------------")
      print("total training epochs " + str(train_epoch))

      summary_writer = tf.summary.FileWriter(self.output_path, graph=sess.graph)
      start_epoch = 0
      if restore:
        ckpt = tf.train.get_checkpoint_state(self.output_path)
        if ckpt and ckpt.model_checkpoint_path:
          saver = tf.train.Saver()
          saver.restore(sess, ckpt.model_checkpoint_path)
          logging.info("model restored from file " + ckpt.model_checkpoint_path)
          start_epoch = int(ckpt.model_checkpoint_path.split("_")[2])

      for epoch in range(start_epoch + 1, train_epoch):

        for step in range(step_size):

          start_index = self.batch_size * step
          end_index = self.batch_size * (step + 1)

          _ = sess.run(self.optimizer,
                       feed_dict={
                         self.network.question: question[start_index: end_index],
                         self.network.answer: answer[start_index: end_index],
                         self.network.answer_mask: answer_mask[
                                                   start_index: end_index],
                         self.network.answer_label: answer_label[start_index: end_index]
                       })

          if step % display_step == 0:

            loss, infer_answers, answers = sess.run([self.network.cost, self.network.infer_outputs,
                                                     self.network.output_tokens],
                                                    feed_dict={
                                                      self.network.question: question[start_index: end_index],
                                                      self.network.answer: answer[start_index: end_index],
                                                      self.network.answer_mask: answer_mask[
                                                                                start_index: end_index],
                                                      self.network.answer_label: answer_label[start_index: end_index]
                                                    })
            logging.info("epoch {:}, step {:}, Minibatch Loss={:.4f}".format(epoch,
                                                                             step,
                                                                             loss))
            summary = tf.Summary()
            summary.value.add(tag="train_loss", simple_value=loss)
            summary_writer.add_summary(summary, epoch * step_size + step)
            summary_writer.flush()
            # decoded_labels = self.decode_answer(answer_label[start_index: end_index])
            # decode_answers = self.decode_answer(answers)
            # decode_infers = self.decode_answer(infer_answers)
            # print(decode_answers[0])
            # print(decode_infers[0])
            # print(decoded_labels[0])

        # shuffle data
        question, answer, answer_mask, answer_label = shuffle(question, answer, answer_mask, answer_label)

        if epoch % save_epoch == 0:
          saver = tf.train.Saver()
          saver.save(sess, self.output_path + "epoch_" + str(epoch))
          logging.info("model saved at epoch " + str(epoch))

        if epoch % self.verify_epoch == 0:

          valid_data = data_processor.get_valid_data()
          valid_question = valid_data["question"]
          valid_answer = valid_data["answer"]
          valid_answer_mask = valid_data["answer_mask"]
          valid_answer_label = valid_data["answer_label"]

          raw_valid_data = data_processor.get_raw_valid()

          valid_step = int(len(valid_question) / self.batch_size)
          total_loss = 0.0
          for step in range(valid_step):
            start_index = self.batch_size * step
            end_index = self.batch_size * (step + 1)

            loss, outputs = sess.run([self.network.cost, self.network.infer_outputs],
                                     feed_dict={
                                       self.network.question: valid_question[start_index: end_index],
                                       self.network.answer: valid_answer[start_index: end_index],
                                       self.network.answer_mask: valid_answer_mask[start_index: end_index],
                                       self.network.answer_label: valid_answer_label[start_index: end_index]
                                     })
            decoded_answers = self.decode_answer(outputs)

            raw_valid_seg = raw_valid_data[start_index: end_index]

            if step == 0 or step == 5:
             print("---------question----------")
             print("".join(raw_valid_data[start_index]["question"]))
             print("--------model output------------")
             print("".join(decoded_answers[0]))
             print("-------ground truth----------")
             decoded_label = self.decode_answer(valid_answer_label[start_index: end_index])
             print("".join(decoded_label[0]))

            for decoded_answer, raw_valid in zip(decoded_answers, raw_valid_seg):
              evaluator.calculate_bleu(decoded_answer, raw_valid["answer"])
            total_loss += loss
          logging.info("loss on valid data " + str(total_loss / valid_step))
          avg_bleu_1, avg_bleu_2, avg_bleu_3, avg_bleu_4 = evaluator.average_bleu()

          logging.info("bleu score 1 " + str(avg_bleu_1))
          logging.info("bleu score 2 " + str(avg_bleu_2))
          logging.info("bleu score 3 " + str(avg_bleu_3))
          logging.info("bleu score 4 " + str(avg_bleu_4))

          valid_summary = tf.Summary()
          valid_summary.value.add(tag="valid_loss", simple_value=(total_loss / valid_step))
          valid_summary.value.add(tag="bleu_1", simple_value=avg_bleu_1)
          valid_summary.value.add(tag="bleu_2", simple_value=avg_bleu_2)
          valid_summary.value.add(tag="bleu_3", simple_value=avg_bleu_3)
          valid_summary.value.add(tag="bleu_4", simple_value=avg_bleu_4)
          summary_writer.add_summary(valid_summary, epoch)
          summary_writer.flush()

  def decode_answer(self, answers):

    decode_answers = []

    for answer in answers:
      decode_answer = []
      for token_id in answer:
        if token_id == 0:
          token = "<pad>"
        else:
          token = self.data_provider.decode_vocab[token_id]
        if token == "<end>" or token == "<pad>":
          break
        decode_answer.append(token)
      decode_answers.append(decode_answer)

    return decode_answers


if __name__ == "__main__":
  data_processor = DataProcessor("./data/QA_data/varicocele/", "./data/QA_data/varicocele/varicocele.json",
                                 word2vec="./data/word2vec/varicocele")
  model = Seq2Seq(data_processor.start_token, data_processor.vocab_embedding)
  trainer = Trainer(model, data_processor, learning_rate=5e-3, batch_size=8)
  trainer.train(train_epoch=100, save_epoch=10, display_step=100, restore=True)
