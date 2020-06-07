import numpy as np
import argparse
from grammar import get_grammar
from semantic_similarity import get_most_similar
from watson_choose import choose_best_emotion


#def driver(input = "my dog died", emotion = "fear", file = "samples.txt", grammar_weight = 1, semantic_weight = 1, sa_weight = 1):

#def driver(output_file, input_file, emotion, grammar_weight, semantic_weight, sa_weight):
def driver(output_file, input, emotion, grammar_weight, semantic_weight, sa_weight):

    # remove words after end of sentence
    clean_up_samples(output_file)

    file = "clean_samples.txt"

    # open samples
    f = open(file)
    lines = f.readlines()
    lines = list(filter(lambda line : line != '\n', lines))


    # get weights
    grammar_result = get_grammar(file)
    grammar_weights =  [grammar_weight * i[0] for i in grammar_result]
    semantic_weights = [i * semantic_weight for i in get_most_similar(input)]
    sa_weights = [i * sa_weight for i in choose_best_emotion(output_file, emotion)]


    norm_sa_weights =  [i/max(sa_weights) for i in sa_weights]
    norm_semantic_weights = [i/max(semantic_weights) for i in semantic_weights]
    norm_grammar_weights = [i/max(grammar_weights) for i in grammar_weights]

    print('\n \n weights : {} .. {} .. {} \n \n'.format(norm_sa_weights, norm_semantic_weights, norm_grammar_weights))


    total = []

    # sum weights
    if len(semantic_weights) != len(grammar_weights) != len(sa_weights):
        print("diff lengths {} ... {} \n".format(semantic_weights, grammar_weights))

    for i in range(len(semantic_weights)):
        total.append(norm_grammar_weights[i] + norm_semantic_weights[i] + norm_sa_weights[i])

    idx = np.argmax(total)

    print("\n --------------- Total Scores: {}".format(total))
    print("\n ---------------- Weights: {} ... {} ... {} \n".format(grammar_weight, semantic_weight, sa_weight))

    print("\n \n ---------------------- index {} ... BEST RESULT {} ----------------------------\n \n".format(idx, lines[idx]))

    return lines[idx]

"""
remove words after end of sentence
"""
def check_sample(line):
        line = line.strip()

        if line.endswith('.'):
            return line
        elif line.endswith('!'):
            return line
        elif line.endswith('?'):
            return line
        else:
            new_line = line.split()
            if len(new_line) == 0:
                return 0
            new_line.pop()
            s = " "
            text = s.join(new_line)
            return check_sample(text)

def clean_up_samples(file):
    file = open(file)
    new_file = open("clean_samples.txt", "w+")

    lines = file.readlines()

    for line in lines:
        cleaned = check_sample(line)
        if cleaned != 0:
            new_file.write(cleaned)
            new_file.write('\n')



# run driver
#driver("samples.txt","my dog dies", "1, 1, 1)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file", type=str, default="samples.txt",
        help="the output of run_pplm"
    )
    parser.add_argument(
        "--input", type=str, default="Mom: Do you want to go on the roller coaster? Son: ",
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
