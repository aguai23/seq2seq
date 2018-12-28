import nltk


class Evaluator(object):

  def __init__(self):

    self.total_bleu = 0.0
    self.sample_number = 0

  def calculate_bleu(self, ref, hyp):
    bleu_score = nltk.translate.bleu_score.sentence_bleu([ref], hyp)
    self.sample_number += 1
    self.total_bleu += bleu_score
    return bleu_score

  def average_bleu(self):
    if self.sample_number == 0:
      raise ValueError("haven't calculate any score ")
    avg_score = self.total_bleu / self.sample_number
    self.total_bleu = 0.0
    self.sample_number = 0
    return avg_score