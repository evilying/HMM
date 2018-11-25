import numpy as np
from utility import add2dict, list2pdf, \
    sample_word, remove_punctuation, gen_tag_dict, \
    get_field, gen_tag_seq, gen_transition
import string, unicodedata, random
import pandas as pd

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

print(transitions_word[word])
print(tag_dict['at'])
print(set(transitions_word[word].keys()) & tag_dict['at'])

tag_seq = ['wql', 'jj', 'at', 'nn', '.']

for tag in tag_seq:

    print(list(tag_dict[tag])[0])
