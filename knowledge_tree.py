# -*- coding: utf-8 -*-
import re
import jieba
import jieba.analyse
import math
import json


class KnowledgeNode(object):

  def __init__(self, entity=None, knowledge=None, parent=None):

    # used for first level node
    self.entity = entity

    # used for second level node
    self.knowledge = knowledge

    # parent
    self.parent = parent

    # children node, may be empty
    self.children = []

    # feature embedding, word2vec or glove
    self.embedding = None

  def add_child(self, child):
    self.children.append(child)
    if child.parent is None:
      child.parent = self


class KnowledgeTree(object):

  def __init__(self, knowledge_file, root_name, dict_path, top_k=5):

    # the knowledge qa pair file
    self.knowledge_file = knowledge_file

    # root is always the disease name
    self.root = KnowledgeNode(entity=[root_name])

    # question list
    self.questions = None

    # answer list, with segmented parts
    self.answers = None

    # document = question + answer
    self.documents = []

    # entity list for each document, used for first level nodes
    self.entities = []

    self.top_k = top_k

    self.dict_path = dict_path

    self.parse_knowledge_file()

    self.extract_entities(dict_path, top_k=top_k)

    self.build_graph()

  def parse_knowledge_file(self):
    """
    we read the gold question answer pairs in, and parse them to structured data
    :return: questions, answers, documents will be filled
    """
    # remove useless characters
    question_regix = re.compile('[？. 0-9（）]')
    answer_regix = re.compile('（[0-9]）|[ 「」\u2028]')

    with open(self.knowledge_file, "r") as f:
      lines = f.readlines()
      questions = []
      answers = []
      answer_parts = []
      for line in lines:
        line = line.strip("\n")
        if len(line) > 0 and line.endswith("？"):
          # print(line)
          if len(answers) < len(questions):
            answers.append(answer_parts)
            answer_parts = []
          questions.append(question_regix.sub('', line))
        elif len(line) > 0:
          answer_parts.append(answer_regix.sub('',line))
      answers.append(answer_parts)
      # print(questions)
      # print(answers)
    assert len(questions) == len(answers)
    # print(len(questions))
    self.questions = questions
    self.answers = answers

    # get document for each question answer pair
    for i in range(len(questions)):
      question = questions[i]
      answer = answers[i]
      document = ""
      document += question
      for answer_part in answer:
        document += answer_part
      self.documents.append(document)

  def extract_entities(self, dict_path=None, top_k=5):
    """
    we use tf-idf to extract important entities for each document
    :param dict_path: used for jieba cut
    :return: entities for each document
    """
    if dict_path is None:
      raise ValueError("must provide dict path")
    jieba.load_userdict(dict_path)

    # count word for each document
    word_sets = []
    for document in self.documents:
      words = jieba.cut(document)
      word_set = {}
      for word in words:
        if word in word_set:
          word_set[word] += 1
        else:
          word_set[word] = 1
      word_sets.append(word_set)

    for word_set in word_sets:
      score_dict = {}
      for word in word_set.keys():
        tf = math.log(1 + word_set[word])
        # tf = 1
        total_document = len(self.documents)
        contained_document = 0.0
        for iter_set in word_sets:
          if word in iter_set:
            contained_document += 1
        idf = math.log(total_document / contained_document)
        score_dict[word] = idf * tf
      sorted_entity = sorted(score_dict.items(), key=lambda kv: kv[1], reverse=True)
      entity_list = []
      for entity in sorted_entity[:top_k]:
        entity_list.append(entity[0])
      self.entities.append(entity_list)
      # print(word_set)
      # print(entity_list)

  def build_graph(self):

    # build first level children
    for entity in self.entities:
      topic_node = KnowledgeNode(entity=entity, parent=self.root)
      self.root.add_child(topic_node)

    # build second level children
    topic_nodes = self.root.children
    for i in range(len(self.answers)):
      knowledge_list = self.answers[i]
      topic_node = topic_nodes[i]
      for knowledge in knowledge_list:
        knowledge_node = KnowledgeNode(knowledge=knowledge, parent=topic_node)
        topic_node.add_child(knowledge_node)
    # print("first level nodes " + str(len(topic_nodes)))
    # for topic_node in topic_nodes:
    #   print(len(topic_node.children), end=" ")

  def extract_embedding(self, vocab, max_sequence=30):

    jieba.load_userdict(self.dict_path)
    for entity_node in self.root.children:
      assert len(entity_node.entity) == self.top_k
      embedding = []
      for item in entity_node.entity:
        if item in vocab:
          embedding.append(vocab[item])
        else:
          embedding.append(vocab["unk"])
      for knowledge_node in entity_node.children:
        words = list(jieba.cut(knowledge_node.knowledge))
        knowledge_node.embedding = self.sentence_to_embedding(words, vocab, max_sequence)
        # print(len(knowledge_node.embedding))
      entity_node.embedding = embedding

  @staticmethod
  def sentence_to_embedding(tokens, vocab, max_sequence):
    if len(tokens) > max_sequence:
      tokens = tokens[:max_sequence]

    embedding = []
    for token in tokens:
      if token in vocab:
        embedding.append(vocab[token])
      else:
        embedding.append(vocab["unk"])

    while len(embedding) < max_sequence:
      embedding.append(vocab["[PAD]"])
    return embedding


if __name__ == "__main__":
  knowledge_tree = KnowledgeTree("./data/diabetes_knowledge.txt", "糖尿病", "./data/dict.txt")
  vocab = json.load(open("./data/vocab.json"))
  knowledge_tree.extract_embedding(vocab)