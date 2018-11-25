import numpy as np
import string
import unicodedata
import sys, random


tbl = dict.fromkeys(i for i in range(sys.maxunicode)
                      if unicodedata.category(chr(i)).startswith('P'))

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


def gen_tag_seq(initial, transitions_pos):

    sentence = []
    w0 = sample_word(initial, 0)
    sentence.append(w0)
    w1 = sample_word(transitions_pos[w0])
    while w1 != 'END':

        sentence.append(w1)
        w1 = sample_word(transitions_pos[w1])

    return sentence

def gen_tag_dict(tag_dict, text):

    for i in range(len(text)):

        tokens = text[i].lower().split()
        for token in tokens:

            word_tag = token.split('/')
            word = word_tag[0]
            tag = word_tag[1]
            if tag not in tag_dict.keys():
                tag_dict[tag] = set()
            tag_dict[tag].add(word)
    return tag_dict

def remove_punctuation(text):

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
