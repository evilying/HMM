import numpy as np
from utility import add2dict, list2pdf, \
    sample_word, remove_punctuation, viterbi, \
    print_optimal_seq
import string
import matplotlib.pyplot as pl

data_dir = './data/'

initial = {}
words = {}
transitions = {}

filename = data_dir + 'robert_frost.txt'
contents = open(filename, 'r', encoding = 'utf-8')

for iline in range(10):

    line = contents.readline()
    tokens = remove_punctuation(line.rstrip().lower()).split()
    print(tokens)
    n_gram = 1
    ntokens = len(tokens)

    for i in range(ntokens):

        current_token = tokens[i]
        add2dict(words, 'words', current_token)
        if i == ntokens-1:
            add2dict(transitions, current_token, 'END')
        else:
            if i == 0:
                initial[current_token] = initial.get(current_token, 0) + 1
            key_token = ' '.join(tokens[i: i + n_gram])
            if key_token not in transitions.keys():
                transitions[key_token] = []
            transitions[key_token].append(tokens[i + n_gram])

initial_total = sum(initial.values())
for token, count in initial.items():

        initial[token] = count / initial_total

for token, wlist in transitions.items():

        transitions[token] = list2pdf(wlist)

words['words'] = list2pdf(words['words'])

word_encode = {}
num_decode = {}
num_code = 0
for word in words['words']:

    word_encode[word] = num_code
    num_decode[num_code] = word
    num_code += 1

word_encode['END'] = num_code
num_decode[num_code] = 'END'
nwords = len(word_encode)
P = np.zeros((nwords, nwords))
for token, wdict in transitions.items():

    print(token, wdict)
    itoken = word_encode[token]
    for next_word, prob in wdict.items():

        print(word_encode[next_word], prob)
        inext = word_encode[next_word]
        P[itoken, inext] = prob
len_path = 20
P_pow = np.zeros((len_path+1, nwords, nwords))
P_pow[0, :, :] = P
for i in range(len_path):
    i += 1
    P_pow[i, :, :] = np.matmul(P_pow[i-1, :, :], P)
word = 'better'
if word not in word_encode.keys():
    print(word, 'not in the dictionary.')
code = word_encode[word]
for ini_word in initial.keys():

    print(ini_word)
    ini_code = word_encode[ini_word]
    for i in range(5):

        print('step', i, P_pow[i, ini_code, code])

k = 3
PI = 3 * np.ones((k+1, nwords, nwords))
u = word_encode['then']
bq = -1 * np.ones_like(PI)
v = word_encode['better']
prob = viterbi(PI, P, k, u, v, nwords, bq, num_decode)
if prob > 0:
    seq = ['better']
    print_optimal_seq(seq, bq, k, u, v, num_decode)
    print(seq[::-1])
else:
    print('no')
