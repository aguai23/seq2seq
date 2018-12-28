import json
import jieba
from random import shuffle
import gensim
import numpy as np
import re


class DataProcessor(object):

  def __init__(self, data_dir, qa_file=None, train_file=None, valid_file=None, test_file=None, word2vec=None,
               train_percent=0.8, train_word2vec=False, max_question_token=30, max_answer_token=100, embedding_size=100):
    self.data_dir = data_dir
    self.train_data = None
    self.valid_data = None
    self.test_data = None

    self.max_question_token = max_question_token
    self.max_answer_token = max_answer_token
    self.embedding_size = embedding_size

    self.start_token = None
    self.end_token = None
    self.unknown_token = None
    self.padding_token = None

    if qa_file is None:
      self.train_data = json.load(open(train_file, "r"))
      self.valid_data = json.load(open(valid_file, "r"))
      self.test_data = json.load(open(test_file, "r"))
    else:
      self.qa_file = qa_file
      self.process_data(train_percent)

    print("training number " + str(len(self.train_data)))
    print("valid number " + str(len(self.valid_data)))
    print("test number " + str(len(self.test_data)))

    if word2vec:
      self.word2vec = gensim.models.Word2Vec.load(word2vec)
      # print(self.word2vec["血管"])
      # print(self.word2vec.most_similar(positive=[self.word2vec["血管"]],negative=[], topn=1))
    else:
      if train_word2vec:
        document = []
        for qa_pair in self.train_data + self.valid_data + self.test_data:
          document.append(qa_pair["question"])
          document.append(qa_pair["answer"])
        model = gensim.models.word2vec.Word2Vec(
          document,
          size=embedding_size,
          window=5,
          min_count=1,
          workers=3
        )
        model.train(document, total_examples=len(document), epochs=30)
        model.save("./data/word2vec/varicocele")
        self.word2vec = model

    self.raw_valid_data = self.valid_data
    self.raw_test_data = self.test_data

    self.encode_vocab, self.decode_vocab = self.build_vocab()
    self.train_data = self.convert_to_embedding(self.train_data)
    self.valid_data = self.convert_to_embedding(self.valid_data)
    self.test_data = self.convert_to_embedding(self.test_data)

    self.vocab_embedding = self.build_vocab_embedding()
    print(self.vocab_embedding.shape)

  def build_vocab_embedding(self):

    vocab_embedding = []
    vocab_embedding.append(self.padding_token)
    vocab_embedding.append(self.unknown_token)
    vocab_embedding.append(self.start_token)
    vocab_embedding.append(self.end_token)
    for index in range(4, len(self.encode_vocab) + 1):
      vocab_embedding.append(self.word2vec[self.decode_vocab[index]])
    return np.asarray(vocab_embedding)

  def build_vocab(self):
    encode_vocab = {}
    decode_vocab = {}
    encode_vocab["unk"] = 1
    encode_vocab["<start>"] = 2
    encode_vocab["<end>"] = 3
    decode_vocab[1] = "unk"
    decode_vocab[2] = "<start>"
    decode_vocab[3] = "<end>"

    index = 4
    for qa_pair in self.train_data + self.valid_data + self.test_data:
      for token in qa_pair["question"] + qa_pair["answer"]:
        if token not in encode_vocab:
          encode_vocab[token] = index
          decode_vocab[index] = token
          index += 1
    print("vocab size " + str(len(encode_vocab)))
    return encode_vocab, decode_vocab

  def get_train_data(self):
    return self.train_data

  def get_valid_data(self):
    return self.valid_data

  def get_raw_valid(self):
    return self.raw_valid_data

  def convert_to_embedding(self, data):
    padding_token = np.zeros(self.embedding_size)
    start_token = np.zeros(self.embedding_size)
    end_token = np.zeros(self.embedding_size)
    unknown_token = np.zeros(self.embedding_size)
    start_token[0] = 1
    end_token[-1] = 1
    unknown_token[0] = 1
    unknown_token[-1] = 1

    data_embedding = []
    for qa_pair in data:
      question = qa_pair["question"]
      answer = qa_pair["answer"]
      if len(question) > self.max_question_token:
        question = question[:self.max_question_token]

      if len(answer) > self.max_answer_token - 2:
        answer = answer[:self.max_answer_token - 2]

      question_embedding = []
      for token in question:
        question_embedding.append(self.word2vec[token])
      while len(question_embedding) < self.max_question_token:
        question_embedding.append(padding_token)

      answer_embedding = [start_token]
      answer_label = []
      for token in answer:
        answer_embedding.append(self.word2vec[token])
        answer_label.append(self.encode_vocab[token])
      token_mask = np.zeros(self.max_answer_token)
      token_mask[:len(answer_embedding)] = 1

      answer_embedding.append(end_token)
      while len(answer_embedding) < self.max_answer_token:
        answer_embedding.append(padding_token)

      while len(answer_label) < self.max_answer_token:
        answer_label.append(0)

      question_embedding = np.asarray(question_embedding)
      answer_embedding = np.asarray(answer_embedding)

      assert question_embedding.shape == (self.max_question_token, self.embedding_size)
      assert answer_embedding.shape == (self.max_answer_token, self.embedding_size)

      data_embedding.append({"question": question_embedding,
                             "answer": answer_embedding,
                             "answer_mask": token_mask,
                             "answer_label": answer_label})

    self.start_token = start_token
    self.end_token = end_token
    self.unknown_token = unknown_token
    self.padding_token = padding_token

    question_embedding = []
    answer_embedding = []
    mask_embedding = []
    label_embedding = []
    for data_item in data_embedding:
      question_embedding.append(data_item["question"])
      answer_embedding.append(data_item["answer"])
      mask_embedding.append(data_item["answer_mask"])
      label_embedding.append(data_item["answer_label"])
    return {
      "question": np.asarray(question_embedding),
      "answer": np.asarray(answer_embedding),
      "answer_mask": np.asarray(mask_embedding),
      "answer_label": np.asarray(label_embedding)
    }

  def process_data(self, train_percent):
    """
    process the qa pair file and split into train, valid ,test
    :param train_percent: training percent
    :return: dump to file
    """

    qa_pairs = json.load(open(self.qa_file, "r"))
    print("total number of data " + str(len(qa_pairs)))

    clean_pairs = []
    for qa_pair in qa_pairs:
      question = qa_pair["question"]
      answer = qa_pair["answer"]
      question = self.clean_string(question)
      answer = self.clean_string(answer)
      clean_pair = {"question": question, "answer": answer}
      clean_pairs.append(clean_pair)

    qa_pairs = clean_pairs
    total_number = len(qa_pairs)
    train_number = int(total_number * train_percent)
    valid_number = int((total_number - train_number) / 2)

    shuffle(qa_pairs)
    self.train_data = qa_pairs[:train_number]
    self.valid_data = qa_pairs[train_number: train_number + valid_number]
    self.test_data = qa_pairs[train_number + valid_number:]

    with open(self.data_dir + "train.json", "w") as f:
      json.dump(self.train_data, f, ensure_ascii=False)

    with open(self.data_dir + "valid.json", "w") as f:
      json.dump(self.valid_data, f, ensure_ascii=False)

    with open(self.data_dir + "test.json", "w") as f:
      json.dump(self.test_data, f, ensure_ascii=False)

  def clean_string(self, string):
    characters_to_remove = {"？", "！", "》", "、", "\r", "%", "&", "*", "…", "（",
                            "@", "#", "￥", "）", ".", ",", " ", "?", "/", "`", "~"}

    clean_string = "".join(c for c in string if c not in characters_to_remove)
    tokens = list(jieba.cut(clean_string))
    clean_tokens = []
    for token in tokens:
      if self.contain_digit(token):
        clean_tokens.append("<number>")
      else:
        clean_tokens.append(token)
    # print(len(clean_tokens))
    return clean_tokens

  @staticmethod
  def contain_digit(phrase):
    return any(c.isdigit() for c in phrase)


if __name__ == "__main__":
  data_processor = DataProcessor("./data/QA_data/varicocele/", "./data/QA_data/varicocele/varicocele.json",
                                 train_word2vec=True)
