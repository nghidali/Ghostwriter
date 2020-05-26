import os
import json
from os.path import join, dirname

import sys

import pathlib

import os

from ibm_watson import ToneAnalyzerV3

from ibm_cloud_sdk_core.authenticators import IAMAuthenticator


api_key = "zMJdye9Ex8FZe1HomlcLkGoVSEuqakSCgB1QYIlcTmPO"


def main():
    in_dial = open('samples.txt', 'r')
    dialogue_tones = []
    for sentence in in_dial:
        tone = get_text_sentiment(sentence)
        dialogue_tones.append(tone)

    print(dialogue_tones)


def get_text_sentiment(text):
    authenticator = IAMAuthenticator(api_key)

    tone_analyzer = ToneAnalyzerV3(version="2019-11-24", authenticator=authenticator)

    tone_analyzer.set_service_url('https://gateway.watsonplatform.net/tone-analyzer/api')

    return tone_analyzer.tone(text).result['document_tone']['tones']

if __name__ == '__main__':
    main()