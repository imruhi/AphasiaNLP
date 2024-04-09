import nltk
import pandas as pd
import numpy as np
import re
from collections import Counter

def tuple_to_str(tuple):
    output_str = ""
    for tup in tuple:
        try:
            output_str += " " + tup[0]
            output_str += " " + tup[1]
        except:
            continue
    return output_str[1:]


def get_single_repetitions(text):
    """
    Return duplicated words (stuttering) and duplicated pauses from utterance.
    e.g: I I I I I I wanted --> I wanted.
    :param text: input text containing dupes.
    :return: a series of the duplicated word and how many times it was repeated.
    """

    utterance = re.findall("[a-zA-Z_]+", text)
    newlist = []
    for i, element in enumerate(utterance):
        if i > 0 and utterance[i - 1] == element and element.isalnum():
            newlist.append(element)
    counts = Counter(newlist)
    if counts:
        for x in counts.keys():
            counts[x] += 1
        return counts
    else:
        return None


def get_bigram_repetitions(text):
    """
    Removes bigram stuttering from text. I went I went to the to the doctor --> I went to the doctor.
    :param text: input text containing a string
    :return: string without duplicates.
    """
    bigram = list(nltk.bigrams(text.split()))
    grams = []

    for i in range(0, len(bigram)):
        if i % 2 == 0:
            grams.append(bigram[i])

    result = []
    prev_item = None
    for item in grams:
        if item != prev_item:
            result.append(item)
            prev_item = item

    if result[-1][-1] != bigram[-1][-1]:
        result.append(tuple((bigram[-1][-1]).split(" ")))

    return tuple_to_str(result)