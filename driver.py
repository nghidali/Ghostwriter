import numpy as np
from grammar import get_grammar
from semantic_similarity import get_most_similar
from watson_choose import choose_best_emotion


def driver(input = "my dog died", emotion = "fear", file = "samples.txt", grammar_weight = 1, semantic_weight = 1, sa_weight = 1):

    # get inputs
    inputs = open("input.txt").readlines()
    input = inputs[0]
    emotion = inputs[1]

    print("input and emotion: {} ...... {} \n".format(input, emotion))

    # open samples
    f = open(file)
    lines = f.readlines()
    lines = list(filter(lambda line : line != '\n', lines))

    print("length {}".format(len(lines)))


    # get weights
    grammar_result = get_grammar(file)
    grammar_weights =  grammar_weight * [i[0] for i in grammar_result]
    semantic_weights = semantic_weight * get_most_similar(input)
    sa_weights = sa_weight * choose_best_emotion(file, emotion)
    total = []

    # sum weights
    if len(semantic_weights) != len(grammar_weights) != len(sa_weights):
        print("diff lengths {} ... {} \n".format(semantic_weights, grammar_weights))

    for i in range(len(semantic_weights)):
        total.append(grammar_weights[i] + semantic_weights[i] + sa_weights[i])

    idx = np.argmax(total)

    print("\n \n ---------------------- index {} ... RESULT {} ----------------------------\n \n".format(idx, lines[idx]))

    return lines[idx]



# run driver
#driver("samples.txt","my dog dies", "1, 1, 1)
