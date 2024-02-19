import spacy
from spacy.matcher import Matcher
import numpy as np
import pandas as pd
from canonical_sents import get_canonical_sentences
import os

nlp = spacy.load("en_core_web_sm")


def aphasic_speech(text, doc):
    vp_pattern = [[{'POS': 'VERB', 'OP': '?'},
                   {'POS': 'ADV', 'OP': '*'},
                   {'POS': 'AUX', 'OP': '*'},
                   {'POS': 'VERB', 'OP': '+'}]]
    matcher = Matcher(nlp.vocab)
    matcher.add("Verb phrase", vp_pattern)
    n = 15
    aphasic_utt = ""

    if len(text.split()) <= n:
        # get NPs
        noun_phrases = set()

        for nc in doc.noun_chunks:
            for nop in [nc, doc[nc.root.left_edge.i:nc.root.right_edge.i + 1]]:
                noun_phrases.add(nop.text.strip())
                # get VPs
        verb_phrases = matcher(doc)
        verb_phrases = [doc[start:end] for _, start, end in verb_phrases]

        try:
            ratio = len(noun_phrases) / len(verb_phrases)
        except:
            # print("Division by zero")
            return aphasic_utt

        X = np.random.uniform(0, 1)

        if ratio > 2 and X <= 0.8:
            # skip sentence
            return aphasic_utt
        else:
            # dont skip sentence
            for tok in doc:
                word_sub = np.random.uniform(0, 1)
                # if word_sub > 0.9:
                # TODO: substitution based on levenshtein distance?
                if tok.dep_ in ["det", "prep", "cop", "aux"]:
                    # determiners, prepositions, copulas
                    Y = np.random.uniform(0, 1)
                    if Y > 0.9:
                        aphasic_utt += tok.text + " "
                elif tok.pos_ in ["ADJ", "ADV"]:
                    # adjectives, adverbs
                    Z = np.random.uniform(0, 1)
                    if Z > 0.5:
                        aphasic_utt += tok.text + " "
                elif tok.pos_ == "VERB":
                    # verbs
                    aphasic_utt += tok.lemma_ + " "
                else:
                    # all other pos
                    aphasic_utt += tok.text + " "

    return aphasic_utt


def augment(filepath, save_path, include_canonical=False):
    texts = pd.read_csv(filepath).reset_index()
    sentences = texts["preprocessed_text"]
    n1 = len(sentences)

    print("Augmenting data")

    if os.path.isfile(save_path):
        df = pd.read_csv(save_path).reset_index()
        print("Data augment exists")
        n2 = len(df["preprocessed_text"])

    else:
        if include_canonical:
            sentences = get_canonical_sentences(sentences, "canonical.csv")
            n2 = len(sentences)
            print(f"Canonical sentences in data: {n2 / n1}")

        augmented_sentences = [aphasic_speech(x, nlp(x)) for x in sentences]
        augmented_sentences = [x.capitalize() for x in augmented_sentences if x is not None and x != '']
        n2 = len(augmented_sentences)
        df = pd.DataFrame({"preprocessed_text": augmented_sentences, "label": 1})
        df.to_csv(save_path, index=False)

    print(f"Sentences retained after augmentation: {n2 / n1}")


