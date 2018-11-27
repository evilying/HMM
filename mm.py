import numpy as np
from utility import add2dict, list2pdf, \
    sample_word, remove_punctuation, viterbi, \
    print_optimal_seq, search_max_len, \
    search_optimal_sent, search
import string
import matplotlib.pyplot as pl

data_dir = './data/'

initial = {}
words = {}
transitions = {}

filename = data_dir + 'robert_frost.txt'
contents = open(filename, 'r', encoding = 'utf-8')

for iline in range(20):

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

    # print(token, wdict)
    itoken = word_encode[token]
    for next_word, prob in wdict.items():

        # print(word_encode[next_word], prob)
        inext = word_encode[next_word]
        P[itoken, inext] = prob
len_path = 20
P_pow = np.zeros((len_path+1, nwords, nwords))
P_pow[0, :, :] = P
for i in range(len_path):
    i += 1
    P_pow[i, :, :] = np.matmul(P_pow[i-1, :, :], P)

current_word = 'i'
opt_seq_first, prob_first = search(initial, current_word, P_pow, P, \
            len_path, word_encode, num_decode)
end_word = 'END'
initial_cur = {}
initial_cur[current_word] = ''
opt_seq_second, prob_sec = search(initial_cur, end_word, P_pow, P, \
            len_path, word_encode, num_decode)

print(type(opt_seq_first), prob_first)
print(type(opt_seq_second[1:]), prob_sec)
opt_seq = opt_seq_first
opt_seq.extend(opt_seq_second[1:])
opt_prob = prob_first * prob_sec
print(opt_seq, opt_prob)
