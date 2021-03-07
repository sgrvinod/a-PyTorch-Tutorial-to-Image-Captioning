import os
from sent2vec.vectorizer import Vectorizer
from scipy import spatial


def compare_two_sentences(sentence_1, sentence_2):
    sentences = [sentence_1, sentence_2]

    vectorizer = Vectorizer()
    vectorizer.bert(sentences)
    vec_1, vec_2 = vectorizer.vectors

    dist = spatial.distance.cosine(vec_1, vec_2)
    return dist
def dir_creator(dir_name):
    # print('checking_dir:', dir_name)
    try:
        tmp_dir = os.getcwd()
        os.chdir(dir_name)
        for _ in dir_name.split('/')[:-1]:
            os.chdir('..')
        os.chdir(tmp_dir)
    except FileNotFoundError:
        if len(dir_name.split('/')) > 1:
            tot_dir = ''
            for dir in dir_name.split('/'):
                tot_dir += dir
                try:os.mkdir(tot_dir)
                except FileExistsError:pass
                tot_dir += '/'
        else:
            os.mkdir(dir_name)
