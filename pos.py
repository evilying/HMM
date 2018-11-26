import numpy as np
from utility import add2dict, list2pdf, \
    sample_word, remove_punctuation, gen_tag_dict, \
    get_field, gen_tag_seq, gen_transition, \
    gen_next_max, gen_next_rand
import string, unicodedata, random
import pandas as pd
import operator

data_dir = './data/'

corpus_words = set()
n_gram = 1
filename = data_dir + 'brown.csv'
corpus = pd.read_csv(filename)
label = 'religion'
field_corpus = corpus[corpus['label'] == label]
tag_dict = {}
word_dict = {}

raw_text = get_field(field_corpus, 'raw_text')
gen_tag_dict(tag_dict, word_dict, raw_text)

tokenized_pos = get_field(field_corpus, 'tokenized_pos')
transitions_pos, initial_pos = gen_transition(tokenized_pos, n_gram)

tokenized_word = get_field(field_corpus, 'tokenized_text')
transitions_word, initial_word = gen_transition(tokenized_word, n_gram)
# print(transitions_word)
word = 'human'
if word in word_dict.keys():
    print('word "', word, '" is in the dictionay.')

pos_word = list(word_dict[word])[0]

tag_seq = []
tag_seq.append(pos_word)
print(pos_word)
# stats = transitions_pos[pos_word]
# print(max(stats, key=lambda key: stats[key]))

next_pos = gen_next_max(pos_word, transitions_pos)
while next_pos not in ['.', 'END'] :

    next_pos = gen_next_rand(next_pos, transitions_pos)
    tag_seq.append(next_pos)
# print(set(transitions_word[word].keys()) & tag_dict['at'])
print(tag_seq)
# tag_seq = ['wql', 'jj', 'at', 'nn', '.']

for tag in tag_seq:

    print(list(tag_dict[tag])[0])
