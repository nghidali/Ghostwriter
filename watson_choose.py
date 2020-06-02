import argparse

import numpy as np

from ibm_watson import ToneAnalyzerV3

from ibm_cloud_sdk_core.authenticators import IAMAuthenticator


api_key = "zMJdye9Ex8FZe1HomlcLkGoVSEuqakSCgB1QYIlcTmPO"

def choose_best_emotion(text_file, emotion):
    target_emotion = emotion
    in_dial = open(text_file, 'r')
    dialogue_tones = []
    sentences = []
    for sentence in in_dial:
        if sentence != '\n':
            tone = get_text_sentiment(sentence)
            dialogue_tones.append(tone)
            sentences.append(sentence)
    best_sentence = -1
    best_score = 0
    scores = np.zeros(len(sentences))
    for sentence in range(len(dialogue_tones)):
        for sentiment in range(len(dialogue_tones[sentence])):
            el = dialogue_tones[sentence][sentiment]
            if el['tone_id'] == target_emotion:
                scores[sentence] = el['score']
                if el['score'] > best_score:
                    best_score = el['score']
                    best_sentence = sentences[sentence]

    print('\n \n The winner is:', best_sentence)
    print('\n \n emotion: {} \n \n'.format(emotion))

    return scores


def get_text_sentiment(text):
    authenticator = IAMAuthenticator(api_key)

    tone_analyzer = ToneAnalyzerV3(version="2019-11-24", authenticator=authenticator)

    tone_analyzer.set_service_url('https://gateway.watsonplatform.net/tone-analyzer/api')

    return tone_analyzer.tone(text).result['document_tone']['tones']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text_file", type=str, default="samples.txt",
        help="the output of run_pplm"
    )
    parser.add_argument(
        "--emotion", type=str, default="joy",
        help="One of: anger, fear, joy, sadness, analytical, confident, tentative"
    )
    args = parser.parse_args()
    choose_best_emotion(**vars(args))
