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
ntokens = len(tokens)
for i in range(ntokens):

    current_token = tokens[i]
    if i == 0:
        initial[current_token] = initial.get(current_token, 0) + 1
    elif i == ntokens-1:
        add2dict(transitions, current_token, 'END')
        pass
    else:
        key_token = ' '.join(tokens[i: i + n_gram])
        if key_token not in transitions.keys():
            transitions[key_token] = []
        transitions[key_token].append(tokens[i + n_gram])

initial_total = sum(initial.values())
for token, count in initial.items():

        initial[token] = count / initial_total

for token, wlist in transitions.items():

        transitions[token] = list2pdf(wlist)

print(transitions)
