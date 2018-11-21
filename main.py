import numpy as np
from utility import add2dict, list2pdf, sample_word

wlist = ['cat', 'cat', 'dog']

dict_pdf = list2pdf(wlist)

print(dict_pdf)

word_initial = sample_word(dict_pdf, 0)

print(word_initial)
