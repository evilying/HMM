import numpy as np
import string
import unicodedata
import sys, random
UPPER = 3
def search_optimal_sent(len_ini, word_ini, \
        end_word, P, nwords, word_encode, num_decode):

    k = len_ini
    # print(k, nwords)
    PI = 3 * np.ones((k+1, nwords, nwords))
    u = word_encode[word_ini]
    bq = -1 * np.ones_like(PI)
    v = word_encode[end_word]
    prob = viterbi(PI, P, k, u, v, nwords, bq, num_decode)
    if prob > 0:
        seq = [end_word]
        print_optimal_seq(seq, bq, k, u, v, num_decode)
        return seq[::-1]
    else:
        return 'cannot compose'

def search_max_len(initial, end_word, P_pow, len_path, word_encode):

    end_code = word_encode[end_word]
    nwords_initial = len(initial.keys())
    step_initial = {}
    word_ini_best = ''
    len_ini_min = 0
    prob_ini_max = 0
    for ini_word in initial.keys():

        ini_code = word_encode[ini_word]
        for i in range(len_path+1):

            access = P_pow[i, ini_code, end_code]
            if access > 0:
                dict = {}
                dict['len'] = i + 1
                dict['prob'] = access
                step_initial[ini_word] = dict
                if prob_ini_max < access:
                    prob_ini_max = access
                    len_ini_min = i + 1
                    word_ini_best = ini_word
                break
    return len_ini_min, word_ini_best, prob_ini_max, step_initial


def print_optimal_seq(seq, bq, k, u, v, num_decode):

    if k == 0:
        return
    w = int(bq[k, u, v])
    seq.append(num_decode[w])
    print_optimal_seq(seq, bq, k-1, u, w, num_decode)

def viterbi(PI, P, k, u, v, nwords, bq, num_decode):

    if k == 0:
        if u == v:
            return 1
        else:
            return 0
    if PI[k, u, v] < UPPER:
        value = PI[k, u, v]
        return PI[k, u, v]
    res = np.zeros(nwords)
    w = np.zeros(nwords)
    sent = []
    for i in range(nwords):

        w = i
        sent.append(num_decode[w])
        res[i] = viterbi(PI, P, k-1, u, w, nwords, bq, num_decode) * P[w, v]
        if P[w, v] > 0 and res[i] > 0:
            print(res[i], P[w, v])
            print(k, num_decode[w], num_decode[v])
    PI[k, u, v] = np.max(res)
    w_max = np.argmax(res)
    bq[k, u, v] = w_max
    return np.max(res)

def get_field(corpus, field):

    return corpus[field].as_matrix()

def gen_transition(tokenized_corpus, n_gram):

    initial = {}
    transitions_pos = {}
    for pos in tokenized_corpus:

        tokens = pos.split()
        ntokens = len(tokens)
        for i in range(ntokens):

            current_token = tokens[i]
            if i == ntokens-1:
                add2dict(transitions_pos, current_token, 'END')
            else:
                if i == 0:
                    initial[current_token] = initial.get(current_token, 0) + 1
                key_token = ' '.join(tokens[i: i + n_gram])
                if key_token not in transitions_pos.keys():
                    transitions_pos[key_token] = []
                transitions_pos[key_token].append(tokens[i + n_gram])
    for token, wlist in transitions_pos.items():

            transitions_pos[token] = list2pdf(wlist)

    return transitions_pos, initial

def gen_next_rand(word, transitions):

    stats = transitions[word]
    return sample_word(stats)

def gen_next_max(word, transitions):

    stats = transitions[word]
    if '.' in stats:
        return '.'

    return max(stats, key=lambda key: stats[key])

def gen_tag_seq(initial, transitions_pos):

    sentence = []
    w0 = sample_word(initial, 0)
    sentence.append(w0)
    w1 = sample_word(transitions_pos[w0])
    while w1 != 'END':

        sentence.append(w1)
        w1 = sample_word(transitions_pos[w1])

    return sentence

def gen_tag_dict(tag_dict, word_dict, text):

    for i in range(len(text)):

        tokens = text[i].lower().split()
        for token in tokens:

            word_tag = token.split('/')
            word = word_tag[0]
            tag = word_tag[1]
            if tag not in tag_dict.keys():
                tag_dict[tag] = set()
            tag_dict[tag].add(word)
            if word not in word_dict.keys():
                word_dict[word] = set()
            word_dict[word].add(tag)

def remove_punctuation(text):

    tbl = dict.fromkeys(i for i in range(sys.maxunicode)
            if unicodedata.category(chr(i)).startswith('P'))
    return text.translate(tbl)

def add2dict(dict, key, value):

    if key not in dict.keys():
        dict[key] = []
    dict[key].append(value)

def list2pdf(wlist):

    dict = {}
    nwords = len(wlist)

    for word in wlist:

        dict[word] = dict.get(word, 0) + 1

    for word, count in dict.items():

        dict[word] = count / nwords

    return dict

def sample_word(dict, random_state=None):

    generator = check_random_state(random_state)
    p0 = generator.random_sample(1)
    # print(p0)
    cumulative = 0
    for word, prob in dict.items():

        cumulative += prob
        if p0 < cumulative:
            return word
    assert(False)

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('{} cannot be used to seed a numpy.random.RandomState'
                     ' instance'.format(seed))
