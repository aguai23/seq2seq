import json
import jieba
from random import shuffle
import numpy as np


class DataProcessor(object):

  def __init__(self, data_file, process_data=True, data_dir=None, train_percent=0.8, vocab=None,
               max_sequence=30, max_answers=5):
    # original qa data
    self.data_file = data_file

    self.train_data = None
    self.valid_data = None
    self.test_data = None

    self.data_dir = data_dir
    self.vocab = vocab

    self.max_sequence = max_sequence
    self.max_answers = max_answers

    if process_data:
      self.process_data(train_percent)
    else:
      self.train_data = json.load(open(self.data_dir + "train.json", "r"))
      self.valid_data = json.load(open(self.data_dir + "valid.json", "r"))
      self.test_data = json.load(open(self.data_dir + "test.json", "r"))

    print("training number " + str(len(self.train_data)))
    print("valid number " + str(len(self.valid_data)))
    print("test number " + str(len(self.test_data)))

    self.data_text = self.train_data + self.valid_data + self.test_data

    self.train_data = self.convert_to_feature(self.train_data)
    print(self.train_data["question"].shape)
    print(self.train_data["answer"].shape)
    print(self.train_data["answer_mask"].shape)
    print(self.train_data["stop_label"].shape)
    self.valid_data = self.convert_to_feature(self.valid_data)

  def get_train_data(self):
    return self.train_data

  def get_valid_data(self):
    return self.valid_data

  def process_data(self, train_percent):

    qa_pairs = json.load(open(self.data_file, "r"))
    print("total number of data " + str(len(qa_pairs)))

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

  def build_vocab(self, output_dir, jieba_dict, external_file=None):

    vocab = {}
    index = 0
    vocab["[PAD]"] = index
    index += 1

    for i in range(30):
      vocab["unused" + str(i + 1)] = index
      index += 1

    vocab["unk"] = index
    index += 1
    vocab["[CLS]"] = index
    index += 1
    vocab["[SEP]"] = index
    index += 1

    jieba.load_userdict(jieba_dict)

    for qa_pair in self.data_text:
      question = qa_pair["question"]
      answer = qa_pair["answer"]
      words = jieba.cut(question + answer)
      for word in words:
        if word not in vocab:
          vocab[word] = index
          index += 1

    if external_file:
      with open(external_file, "r") as f:
        lines = f.readlines()
        for line in lines:
          words = jieba.cut(line)
          for word in words:
            if word not in vocab:
              vocab[word] = index
              index += 1

    print("vocab size " + str(len(vocab)))
    json.dump(vocab, open(output_dir + "vocab.json", "w"), ensure_ascii=False)

  def convert_to_feature(self, data):
    if not self.vocab:
      raise ValueError("no vocab")
    vocab = json.load(open(self.vocab, "r"))

    question_embeddings = []
    answer_embeddings = []
    answer_masks = []
    stop_labels = []
    # min_answers = 100
    # max_answers = 0
    # avg_answers = 0.0

    for qa_pair in data:
      question = qa_pair["question"]
      answer = qa_pair["answer"]
      answer_segs = answer.split("。")

      question_tokens = list(jieba.cut(question))
      question_embedding = self.sentence_to_embedding(question_tokens, vocab)
      question_embeddings.append(question_embedding)

      padding_seg = np.zeros(self.max_sequence)
      if len(answer_segs) > self.max_answers:
        answer_segs = answer_segs[:self.max_answers]

      answer_mask = []
      stop_label = []
      answer_embedding = []

      for answer_seg in answer_segs:
        answer_tokens = list(jieba.cut(answer_seg))
        answer_embedding.append(self.sentence_to_embedding(answer_tokens, vocab))
        answer_mask.append(1)
        stop_label.append(0)

      stop_label[-1] = 0

      while len(answer_embedding) < self.max_answers:
        answer_embedding.append(padding_seg)
        answer_mask.append(0)
        stop_label.append(0)

      answer_embeddings.append(answer_embedding)
      answer_masks.append(answer_mask)
      stop_labels.append(stop_label)
      # avg_answers += len(answer_segs)
      # if len(answer_segs) < min_answers:
      #   min_answers = len(answer_segs)
      # if len(answer_segs) > max_answers:
      #   max_answers = len(answer_segs)

    # print("max answers " + str(max_answers))
    # print("min answers " + str(min_answers))
    # print("avg answers " + str(avg_answers / len(data)))
    return {"question": np.asarray(question_embeddings),
            "answer": np.asarray(answer_embeddings),
            "answer_mask": np.asarray(answer_masks, dtype=np.float32),
            "stop_label": np.asarray(stop_labels, dtype=np.float32)}

  def sentence_to_embedding(self, tokens, vocab):
      if len(tokens) > self.max_sequence - 2:
         cut_tokens = tokens[:(self.max_sequence - 2)]
      else:
        cut_tokens = tokens

      embedding = [vocab["[CLS]"]]
      for token in cut_tokens:
        if token in vocab:
          embedding.append(vocab[token])
        else:
          embedding.append(vocab["unk"])
      embedding.append(vocab["[SEP]"])
      while len(embedding) < self.max_sequence:
        embedding.append(vocab["[PAD]"])
      return embedding


if __name__ == "__main__":

  data_processor = DataProcessor("./data/diabetes.json", data_dir="./data/", vocab="./data/vocab.json")
  data_processor.build_vocab("./data/", "./data/dict.txt", external_file="./data/diabetes_knowledge.txt")