import spacy
from spacy.matcher import Matcher
import numpy as np
import pandas as pd
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
import re
from preprocess import preprocess, postprocess

nlp = spacy.load("en_core_web_sm")
ds = load_dataset("imdb")


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

        if ratio > 2 and X > 0.8:
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


def misra_augment(save_path, include_canonical=False):
    texts = ds["train"]["text"]
    sents = []

    for text in texts:
        t = re.sub(r'\<.*?\>', " ", text)
        sentences = sent_tokenize(t)
        for sent in sentences:
            sent = preprocess(sent)
            sents.append(" ".join(sent.split()))

    broca_ds = sents[500:1000]
    control_ds = sents[1000:1500]

    print("Augmenting data")

    augmented_sentences = []
    control_sentences = []

    for x in broca_ds:
        broca_sentence = aphasic_speech(x, nlp(x))
        if isinstance(broca_sentence, str):
            broca_sentence = postprocess(broca_sentence)
            if broca_sentence:
                augmented_sentences.append(broca_sentence)

    for x in control_ds:
        control_sentence = postprocess(x)
        if control_sentence:
            control_sentences.append(control_sentence)

    broca_data = pd.DataFrame(data={"modified": augmented_sentences, "label": [1] * len(augmented_sentences)})
    control_data = pd.DataFrame(data={"modified": control_sentences, "label": [0] * len(control_sentences)})
    data_full_scenario = pd.concat([broca_data, control_data], ignore_index=True)
    data_full_scenario = data_full_scenario.sample(frac=1).reset_index(drop=True)
    data_full_scenario.to_csv(save_path, sep=",", index=False)

    print(f"Sentences retained after augmentation: {len(sents) / len(augmented_sentences)}")
