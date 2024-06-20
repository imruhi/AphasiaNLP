import evaluate
import pandas as pd
import re
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch
from gensim.models import Word2Vec
import numpy as np
import pickle
import evaluate

metric = evaluate.load("accuracy")


def preprocess(filepath):
    df = pd.read_csv(filepath).dropna().reset_index()
    sentences = df["preprocessed_text"]
    sentences = [re.sub(r'\s([?.!"](?:\s|$))', r'\1', x).strip() for x in sentences]
    labels = [1] * len(sentences)  # test for synthetic broca data
    return sentences, labels


def bert_classification(model_path, sentences, labels):
    tokenizer = AutoTokenizer.from_pretrained(model_path, return_tensor="pt")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model = model.to(device)

    pred_labels_bert = []

    for sentence in sentences:
        inputs = tokenizer(sentence, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            logits = model(inputs).logits
        predicted_class_id = logits.argmax().item()
        pred_labels_bert.append(predicted_class_id)

    print(f"BERT Accuracy: {metric.compute(predictions=pred_labels_bert, references=labels)}")


def avg_w2vec(sentences, labels, w2v_model, vocab):
    """
    Average Word2Vec approach for creating a vector for a given sentence from the word embeddings of each words of the sentence.
    """

    transformed = []
    lab = []
    for sentence, label in zip(sentences, labels):
        count = 0
        vector = np.zeros(300)
        for word in sentence.split():
            if word in vocab:
                vector += w2v_model.wv.get_vector(word)
                count += 1
        if count != 0:
            vector /= count
            transformed.append(vector)
            lab.append(label)
    return np.array(transformed), np.array(lab)


def knn_classification(model_path, vocab_path, sentences, labels):
    w2v_model = Word2Vec.load(vocab_path)
    vocab = list(w2v_model.wv.key_to_index.keys())
    loaded_model = pickle.load(open(model_path, 'rb'))

    transformed_text, labels1 = avg_w2vec(sentences, labels, w2v_model, vocab)

    pred_labels_knn = loaded_model.predict(transformed_text)

    # TODO: check if every word has an representation, currently removes sentences where representations of words are
    #  missing
    print(f"KNN accuracy: {metric.compute(predictions=pred_labels_knn, references=[1] * len(pred_labels_knn))}")


if __name__ == "__main__":

    bert_path = "models/bert_finetuned_old"
    knn_path = "models/knn_pickle"
    knn_vocab_path = "models/word2vec_knn.model"

    # filepath1 = "control_aug.csv"
    # filepath2 = "canonical_control_aug.csv"

    # sentences1, labels1 = preprocess(filepath1)
    # sentences2, labels2 = preprocess(filepath2)
    #
    # print("Control augmented")
    # bert_classification(bert_path, sentences1, labels1)
    # knn_classification(knn_path, knn_vocab_path, sentences1, labels1)
    #
    # print("\nControl augmented + canonical sentences")
    # bert_classification(bert_path, sentences2, labels2)
    # knn_classification(knn_path, knn_vocab_path, sentences2, labels2)

    filepath = "test.csv"
    sentences, labels = preprocess(filepath)
    print("Adaptation theory")
    bert_classification(bert_path, sentences, labels)
    knn_classification(knn_path, knn_vocab_path, sentences, labels)

