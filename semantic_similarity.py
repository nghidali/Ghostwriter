"""

using code from https://nlpforhackers.io/wordnet-sentence-similarity/

"""


from nltk.corpus import wordnet as wn
from nltk import word_tokenize, pos_tag
import numpy as np



def penn_to_wn(tag):
    #Convert between a Penn Treebank tag to a simplified Wordnet tag
    if tag.startswith('N'):
        return 'n'

    if tag.startswith('V'):
        return 'v'

    if tag.startswith('J'):
        return 'a'

    if tag.startswith('R'):
        return 'r'

    return None

def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None

    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None


def sentence_similarity(sentence1, sentence2):
    #compute the sentence similarity using Wordnet
    # Tokenize and tag
    sentence1 = pos_tag(word_tokenize(sentence1))
    sentence2 = pos_tag(word_tokenize(sentence2))

    #print(sentence2)

    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in sentence1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in sentence2]

    #print("2 {}".format(synsets2))

    # Filter out the Nones
    print('before')
    print(synsets1)
    print(synsets2)

    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]


    score, count = 0.0, 0

    # For each word in the first sentence
    for synset in synsets1:

        # Get the similarity value of the most similar word in the other sentence
        scores = [synset.path_similarity(ss) for ss in synsets2]
        scores = [s for s in scores if s]

        if len(scores) == 0:
            return 0

        best_score = max(scores)


        # Check that the similarity could have been computed
        if best_score is not None:
            score += best_score
            count += 1

    # Average the values
    if count == 0:
        return 0

    score /= count
    return score


def get_most_similar(focus_sentence):
    f = open("samples.txt")

    similarities = [] # holds all the similarity percentages that we calculate
    sentences = []

    for line in f:

        similarities.append(sentence_similarity(focus_sentence, line))
        sentences.append(line)

        idx = np.argmax(similarities)

    print("\n \n ------------- most similar: {} ------------".format(sentences[idx]))
