import os
import json
from os.path import join, dirname

import sys

import pathlib

import os

from ibm_watson import ToneAnalyzerV3

from ibm_cloud_sdk_core.authenticators import IAMAuthenticator


api_key = "zMJdye9Ex8FZe1HomlcLkGoVSEuqakSCgB1QYIlcTmPO"

example_tones = [[{'score': 0.946588, 'tone_id': 'joy', 'tone_name': 'Joy'}], [{'score': 0.948464, 'tone_id': 'joy', 'tone_name': 'Joy'}, {'score': 0.814556, 'tone_id': 'tentative', 'tone_name': 'Tentative'}], [{'score': 0.984756, 'tone_id': 'tentative', 'tone_name': 'Tentative'}],[{'score': 0.696539, 'tone_id': 'joy', 'tone_name': 'Joy'}], [{'score': 0.807813, 'tone_id': 'joy', 'tone_name': 'Joy'}], [{'score': 0.937679, 'tone_id': 'joy', 'tone_name': 'Joy'}, {'score': 0.624321, 'tone_id': 'analytical', 'tone_name': 'Analytical'}, {'score': 0.655928, 'tone_id': 'tentative', 'tone_name': 'Tentative'}], [{'score': 0.659731, 'tone_id': 'sadness', 'tone_name': 'Sadness'}], [{'score': 0.617241, 'tone_id': 'joy', 'tone_name': 'Joy'}, {'score': 0.973325, 'tone_id': 'tentative', 'tone_name': 'Tentative'}, {'score': 0.832924, 'tone_id': 'analytical', 'tone_name': 'Analytical'}],  [{'score': 0.930272, 'tone_id': 'analytical', 'tone_name': 'Analytical'}, {'score': 0.61457, 'tone_id': 'confident', 'tone_name': 'Confident'}],  [{'score': 0.601071, 'tone_id': 'joy', 'tone_name': 'Joy'}, {'score': 0.722847, 'tone_id': 'tentative', 'tone_name': 'Tentative'}]]

def main():
    target_emotion = 'joy'
    in_dial = open('samples.txt', 'r')
    dialogue_tones = []
    sentences = []
    for sentence in in_dial:
        if sentence != '\n':
            tone = get_text_sentiment(sentence)
            dialogue_tones.append(tone)
            sentences.append(sentence)
    best_sentence = -1
    best_score = 0
    for sentence in range(len(dialogue_tones)):
        for sentiment in range(len(dialogue_tones[sentence])):
            el = dialogue_tones[sentence][sentiment]
            if el['tone_id'] == target_emotion:
                print(sentences[sentence], el['score'])
                if el['score'] > best_score:
                    best_score = el['score']
                    best_sentence = sentences[sentence]

    print('The winner is:', best_sentence)


def get_text_sentiment(text):
    authenticator = IAMAuthenticator(api_key)

    tone_analyzer = ToneAnalyzerV3(version="2019-11-24", authenticator=authenticator)

    tone_analyzer.set_service_url('https://gateway.watsonplatform.net/tone-analyzer/api')

    return tone_analyzer.tone(text).result['document_tone']['tones']

if __name__ == '__main__':
    main()