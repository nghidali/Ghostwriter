from grammarbot import GrammarBotClient

'''
This program uses the grammarbot API
To install, just run 'pip install grammarbot'


To run the program, uncomment the last line at the bottom
'''

# Driver program
# Call this program with a specified file
# It will output an array of the scores and booleans for each line
# The index of each score matches the ordering of the lines
# The boolean indicates whether the line is complete or not based on punctuation
def get_grammar(file):
    client = GrammarBotClient()
    print("\n scores : {} \n".format(scoreAll(file, client)))
    return scoreAll(file, client)


# Calls the score function on each line that gets read from the file
# A little preprocessing done to clean out extra lines
# Stores each score in an array
def scoreAll(file, client):
    lines = open(file).readlines()
    #print("length: {}".format(len(lines)))

    lines = list(filter(lambda line : line != '\n', lines))
    #print("length: {}".format(len(lines)))

    scores = []
    for line in lines:
        scores.append(score(line, client))

    return scores


# Given a line, counts how many grammatical errors appear in the line
# If the line doesn't end with a standard punctation mark, the number of errors are increased by 10
# The complete flag will also be set to false for incomplete responses
# The calculated score is the reciprocal of the number of errors and the number of runs needed to correct
# That means the fewer the errors the higher the score and vice versa
def score(line, client):
    errors = 0
    runs = 1

    line = line.strip(' \n')



    complete = True
    # Sometimes it takes multiple tries to find all the errors
    # The API has an automatic correction agent that will change one error at a time
    # The total result will become the max errors at any point + the number of corrections it took to correct all of them
    while runs < 50:
        if line == '\n':
            break
        response = client.check(line).matches

        e = len(response)
        errors = max(e, errors)

        if e == 0:
            break

        line = response[0].corrections[0]


        runs += 1

    if line[-1] != '.' and line[-1] != '!' and line[-1] != '?':
        errors += 10
        complete = False

    return 1 / (runs + errors), complete




# UNCOMMENT THIS TO RUN
# NEED TO FILL IN THE ARGUMENT FOR THE FILE PATH


#main("samples.txt")
