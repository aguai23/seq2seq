import nltk


class Evaluator(object):

  def __init__(self):

    self.total_bleu_1 = 0.0
    self.total_bleu_2 = 0.0
    self.total_bleu_3 = 0.0
    self.total_bleu_4 = 0.0
    self.sample_number = 0

  def calculate_bleu(self, ref, hyp):
    bleu_1 = nltk.translate.bleu_score.sentence_bleu([ref], hyp, weights=[1, 0, 0, 0])
    bleu_2 = nltk.translate.bleu_score.sentence_bleu([ref], hyp, weights=[0, 1, 0, 0])
    bleu_3 = nltk.translate.bleu_score.sentence_bleu([ref], hyp, weights=[0, 0, 1, 0])
    bleu_4 = nltk.translate.bleu_score.sentence_bleu([ref], hyp, weights=[0, 0, 0, 1])
    self.sample_number += 1
    self.total_bleu_1 += bleu_1
    self.total_bleu_2 += bleu_2
    self.total_bleu_3 += bleu_3
    self.total_bleu_4 += bleu_4

  def average_bleu(self):
    if self.sample_number == 0:
      raise ValueError("haven't calculate any score ")
    avg_score_1 = self.total_bleu_1 / self.sample_number
    avg_score_2 = self.total_bleu_2 / self.sample_number
    avg_score_3 = self.total_bleu_3 / self.sample_number
    avg_score_4 = self.total_bleu_4 / self.sample_number
    self.total_bleu_1 = 0.0
    self.total_bleu_2 = 0.0
    self.total_bleu_3 = 0.0
    self.total_bleu_4 = 0.0
    self.sample_number = 0
    return avg_score_1, avg_score_2, avg_score_3, avg_score_4