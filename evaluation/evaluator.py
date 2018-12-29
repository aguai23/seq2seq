import nltk
from evaluation.cider_scorer import CiderScorer


class Evaluator(object):

  def __init__(self):

    self.total_bleu_1 = 0.0
    self.total_bleu_2 = 0.0
    self.total_bleu_3 = 0.0
    self.total_bleu_4 = 0.0
    self.rouge = 0.0
    self.bleu_number = 0
    self.rouge_number = 0
    self.cider_scorer = CiderScorer()

  def calculate_bleu(self, ref, hyp):
    bleu_1 = nltk.translate.bleu_score.sentence_bleu([ref], hyp, weights=[1, 0, 0, 0])
    bleu_2 = nltk.translate.bleu_score.sentence_bleu([ref], hyp, weights=[0, 1, 0, 0])
    bleu_3 = nltk.translate.bleu_score.sentence_bleu([ref], hyp, weights=[0, 0, 1, 0])
    bleu_4 = nltk.translate.bleu_score.sentence_bleu([ref], hyp, weights=[0, 0, 0, 1])
    self.bleu_number += 1
    self.total_bleu_1 += bleu_1
    self.total_bleu_2 += bleu_2
    self.total_bleu_3 += bleu_3
    self.total_bleu_4 += bleu_4

  def average_bleu(self):

    if self.bleu_number == 0:
      raise ValueError("haven't calculate any score ")

    avg_score_1 = self.total_bleu_1 / self.bleu_number
    avg_score_2 = self.total_bleu_2 / self.bleu_number
    avg_score_3 = self.total_bleu_3 / self.bleu_number
    avg_score_4 = self.total_bleu_4 / self.bleu_number

    self.total_bleu_1 = 0.0
    self.total_bleu_2 = 0.0
    self.total_bleu_3 = 0.0
    self.total_bleu_4 = 0.0
    self.bleu_number = 0

    return avg_score_1, avg_score_2, avg_score_3, avg_score_4

  def calculate_rouge(self, ref, hyp, beta=1.2):
    """
    calculate rouge-L
    :param ref: ground truth
    :param hyp: the predicted output
    :return:
    """
    lcs = self.my_lcs(ref, hyp)
    precision = lcs / float(len(hyp))
    recall = lcs / float(len(ref))

    if precision != 0 and recall != 0:
      score = ((1 + beta**2) * precision * recall)/float(recall + beta**2 * precision)
    else:
      score = 0.0
    self.rouge += score
    self.rouge_number += 1

  def avg_rouge(self):
    if self.rouge_number == 0:
      raise ValueError("no rouge calculated ")

    avg_rouge = self.rouge / self.rouge_number
    self.rouge_number = 0
    self.rouge = 0.0

    return avg_rouge

  @staticmethod
  def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings
    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if len(string) < len(sub):
      sub, string = string, sub

    lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
      for i in range(1, len(string) + 1):
        if string[i - 1] == sub[j - 1]:
          lengths[i][j] = lengths[i - 1][j - 1] + 1
        else:
          lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[len(string)][len(sub)]

  def add_cider_pair(self, pair):
    self.cider_scorer.append_pair(pair)

  def avg_cider(self):
    score, _ = self.cider_scorer.compute_score()
    self.cider_scorer = CiderScorer()
    return score