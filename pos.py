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

raw_text = get_field(field_corpus, 'raw_text')
tag_dict = gen_tag_dict(tag_dict, raw_text)
tokenized_pos = get_field(field_corpus, 'tokenized_pos')

transitions_pos, initial = gen_transition(tokenized_pos, n_gram)
tag_seq = ['wql', 'jj', 'at', 'nn', '.']

for tag in tag_seq:

    print(list(tag_dict[tag])[0])
