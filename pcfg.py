from queue import Queue
from nltk import Tree
from collections import defaultdict

class PCFG:

  def __init__(self, parsed_sentences):
    self.parsed_sentences = parsed_sentences
    self.rules = defaultdict(lambda: 0.0)
    self.symbol_frequencies = defaultdict(lambda: 0.0)

  def generate_rules(self):
    for parsed_sentence in self.parsed_sentences:
      q = Queue()
      q.put(parsed_sentence)
      while not q.empty():
        actual_node = q.get()
        if type(actual_node) is Tree:
          right_side = []
          for children in actual_node:
            q.put(children)
            right_side.append(children.label() if type(children) is Tree else children)
          self.rules[(actual_node.label(), tuple(right_side))] += 1.0
          self.symbol_frequencies[actual_node.label()] += 1.0

  def generate_probabilities(self):
    for k, v in self.rules.items():
      self.rules[k] = v / self.symbol_frequencies[k[0]]

  def run(self):
    self.generate_rules()
    self.generate_probabilities()


if __name__== '__main__':
  from nltk.corpus import floresta
  pcfg = PCFG(floresta.parsed_sents())
  pcfg.run()
  print(pcfg.rules)

