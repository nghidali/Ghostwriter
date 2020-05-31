import numpy as np
import argparse
from grammar import get_grammar
from semantic_similarity import get_most_similar
from watson_choose import choose_best_emotion


#def driver(input = "my dog died", emotion = "fear", file = "samples.txt", grammar_weight = 1, semantic_weight = 1, sa_weight = 1):

def driver(output_file, input_file, emotion, grammar_weight, semantic_weight, sa_weight):
    # get inputs
    inputs = open(input_file).readlines()
    input = inputs[0]
    emotion = inputs[1]

    print("input and emotion: {} ...... {} \n".format(input, emotion))

    # open samples
    f = open(output_file)
    lines = f.readlines()
    lines = list(filter(lambda line : line != '\n', lines))

    print("length {}".format(len(lines)))


    # get weights
    grammar_result = get_grammar(output_file)
    grammar_weights =  [grammar_weight * i[0] for i in grammar_result]
    semantic_weights = [i * semantic_weight for i in get_most_similar(input)]
    sa_weights = [i * sa_weight for i in choose_best_emotion(output_file, emotion)]
    total = []

    # sum weights
    if len(semantic_weights) != len(grammar_weights) != len(sa_weights):
        print("diff lengths {} ... {} \n".format(semantic_weights, grammar_weights))

    for i in range(len(semantic_weights)):
        total.append(grammar_weights[i] + semantic_weights[i] + sa_weights[i])

    idx = np.argmax(total)

    print("\n --------------- Scores: {}".format(total))
    print("\n ---------------- Weights: {} ... {} ... {}".format(grammar_weight, semantic_weight, sa_weight))

    print("\n \n ---------------------- index {} ... RESULT {} ----------------------------\n \n".format(idx, lines[idx]))

    return lines[idx]



# run driver
#driver("samples.txt","my dog dies", "1, 1, 1)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file", type=str, default="samples.txt",
        help="the output of run_pplm"
    )
    parser.add_argument(
        "--input_file", type=str, default="input.txt",
        help="the input of run_pplm"
    )
    parser.add_argument(
        "--emotion", type=str, default="joy",
        help="One of: anger, fear, joy, sadness, analytical, confident, tentative"
    )
    parser.add_argument(
        "--grammar_weight", type=float, default= 1,
        help="weight of grammarbot"
    )
    parser.add_argument(
        "--semantic_weight", type=float, default=0.9,
        help="weight of semantic similarity"
    )
    parser.add_argument(
        "--sa_weight", type=float, default=0.8,
        help="weight of sentiment analysis"
    )
    args = parser.parse_args()
    driver(**vars(args))
