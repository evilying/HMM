import numpy as np
from utility import add2dict, list2pdf, sample_word, remove_punctuation
import string, unicodedata

data_dir = './data/'

initial = {}
second_word = {}
transitions = {}

filename = data_dir + 'robert_frost.txt'
contents = open(filename, 'r', encoding = 'utf-8')

line = contents.readline()
print(line)
tokens = remove_punctuation(line.rstrip().lower()).split()
n_gram = 1
word_grams = {}
for i in range(len(tokens)-n_gram):

    gram = ' '.join(tokens[i: i + n_gram])
    if gram not in word_grams.keys():
        word_grams[gram] = []
    word_grams[gram].append(tokens[i + n_gram])
print(word_grams)
