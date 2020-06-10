"""

using code from https://nlpforhackers.io/wordnet-sentence-similarity/

"""


from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

def get_pos(sentence):
    stop_words = set(stopwords.words('english'))

    tokenizer = RegexpTokenizer(r'\w+')
    word_tokens = tokenizer.tokenize(sentence)

    filtered_sentence = pos_tag([w for w in word_tokens if not w in stop_words])

    results = []

    for item in filtered_sentence:
        start = item[1][0].lower()
        if start == 'n':
            results.append((item[0], start))
        if start == 'v':
            results.append((item[0], start))

    return results

def get_synsets(sentence):
    synsets_lst = []
    for item in sentence:
        if len(item) > 1:
            synset = wn.synsets(item[0], item[1])
            if len(synset) > 0:
                synsets_lst.append(synset[0])

    return synsets_lst


def get_similarity(sent1, sent2):
    sentence1 = get_pos(sent1)
    sentence2 = get_pos(sent2)

    synsets1 = get_synsets(sentence1)
    synsets2 = get_synsets(sentence2)

    total = 0
    count = 0

    for word in synsets1:

        similarity = ([word.path_similarity(s) for s in synsets2])

        # remove Nones
        similarity = [i for i in similarity if isinstance(i, (int, float))]

        total += sum(similarity)
        count += 1


    return total/count

"""
run program
"""

def get_most_similar(focus_sentence):
    f = open("samples.txt")
    lines = f.readlines()
    i = 1

    similarities = [] # holds all the similarity percentages that we calculate
    sentences = []

    for line in lines:
        if line[0] != '\n':
            similarities.append(get_similarity(focus_sentence, line))
            sentences.append(line)
            i+=1

    idx = np.argmax(similarities)

    return similarities # return list
