# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 13:55:21 2018

@author: leock
"""

import gensim
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from math import log
import math
import re
import nltk
from nltk.stem import WordNetLemmatizer
import operator

wordlemmatizer = WordNetLemmatizer()


def freq(words):
    words = [word.lower() for word in words]
    dict_freq = {}
    words_unique = []
    for word in words:
        if word not in words_unique:
            words_unique.append(word)
    for word in words_unique:
        dict_freq[word] = words.count(word)
    return dict_freq


def tf_score(word, sentence):
    freq_sum = 0
    word_frequency_in_sentence = 0
    len_sentence = len(sentence)
    for word_in_sentence in sentence.split():
        if word == word_in_sentence:
            word_frequency_in_sentence = word_frequency_in_sentence + 1
    tf = word_frequency_in_sentence / len_sentence
    return tf


def idf_score(no_of_sentences, word, sentences):
    no_of_sentence_containing_word = 0
    for sentence in sentences:
        sentence = remove_special_characters(str(sentence))
        sentence = re.sub(r'\d+', '', sentence)
        sentence = sentence.split()
        sentence = [word for word in sentence if len(word) > 1]
        sentence = [word.lower() for word in sentence]
        sentence = [wordlemmatizer.lemmatize(word) for word in sentence]
        if word in sentence:
            no_of_sentence_containing_word = no_of_sentence_containing_word + 1
    idf = math.log10(no_of_sentences / no_of_sentence_containing_word)
    return idf


def tf_idf_score(tf, idf):
    return tf * idf


def word_tfidf(dict_freq, word, sentences, sentence):
    word_tfidf = []
    tf = tf_score(word, sentence)
    idf = idf_score(len(sentences), word, sentences)
    tf_idf = tf_idf_score(tf, idf)
    return tf_idf


def remove_special_characters(text):
    regex = r'[^a-zA-Z0-9\s]'
    text = re.sub(regex, '', text)
    return text


def sentence_importance(sentence, dict_freq, sentences):
    sentence_score = 0
    sentence = remove_special_characters(str(sentence))
    sentence = re.sub(r'\d+', '', sentence)
    pos_tagged_sentence = []
    no_of_sentences = len(sentences)
    pos_tagged_sentence = pos_tagging(sentence)
    for word in pos_tagged_sentence:
        if len(word) > 1:
            word = word.lower()
            word = wordlemmatizer.lemmatize(word)
            sentence_score = sentence_score + word_tfidf(dict_freq, word, sentences, sentence)
    return sentence_score


def pos_tagging(text):
    pos_tag = nltk.pos_tag(text.split())
    pos_tagged_noun_verb = []
    for word, tag in pos_tag:
        if tag == "NN" or tag == "NNP" or tag == "NNS" or tag == "VB" or tag == "VBD" or tag == "VBG" or tag == "VBN" or tag == "VBP" or tag == "VBZ":
            pos_tagged_noun_verb.append(word)
    return pos_tagged_noun_verb


def embed_sentences(data, word2vec_limit=50000, NUM_WORDS=20000):
    '''
    Embed sentences

    Params:
        data             - np.array  [ doc id, sentences, saliency score ]

        word2vec_limit   - int: number of words used in the word embedding provided by Google
                            - ex: 50000
        NUM_WORDS        - int: The maximum number of words to keep, based on word frequency. Only the most common num_words words will be kept.
                            - ex: 20000

    Returns:
        input_output        - np.array [embedding matrix , saliency score]

    '''

    # setences ex: ["It's the first sentence!","it is the second sentence"]
    sentences = data[:, 1]

    # Load Google pre-trained words as a model
    embedding_model = gensim.models.KeyedVectors.load_word2vec_format(
        '/content/extractive-document-summarization/word2vec/bert-wordvecs/output.wv', binary=False,
        limit=word2vec_limit)
    # Convert the model as a dictionnary word_vectors["hello"] will return a vector like [0.3, 3, ... , -4]
    word_vectors = embedding_model.wv
    # print("Embedding for 'hello': ", word_vectors["hello"], "\n")

    # Tokenize the sentences, that is to say convert the 2 sentences ["It's the first sentence!","It is the second sentence"] to 2 sequences [[1 4 2 5 3],[1 6 2 7 3]]
    # It handles the ennoying cases (punctuation, Upper cases, etc...)
    tokenizer = Tokenizer(num_words=NUM_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'', lower=True)
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences)
    # print("Padded_sequences: ", padded_sequences, "\n")

    # Produce a dictionnary mapping words to tokens e.g. {'it': 1, 'the': 2, 'sentence': 3, 's': 4, 'first': 5, 'is': 6, 'second': 7}
    word_index = tokenizer.word_index
    # print("word_index: " , word_index , "\n" )

    # start
    word_freq = freq(sentences)
    # c = 1
    # for token in sentences:
    #     sentenceimp = sentence_importance(token, word_freq, sentences)
    #     sentence_with_importance[c] = sentenceimp
    #     c = c + 1
    # sentence_with_importance = sorted(sentence_with_importance.items(), key=operator.itemgetter(1), reverse=True)
    # finish

    # Build a dictionnary mapping tokens to vectors e.g. {'1': [2, ... , -3] ; '2': [4, ... , 0.8] ; ... }
    embedding_weights = {key: embedding_model[word] if word in word_vectors.vocab else
    np.random.uniform(-0.25, 0.25, word_vectors.vector_size) + sentence_importance(sequences, word_freq, sentences)
                         for word, key in word_index.items()}
    # Add the token "0", used for padding
    embedding_weights[0] = np.zeros(word_vectors.vector_size)

    # print("Embedding weights: " , embedding_weights , "\n")

    # Build a 3D array: 1D for the sentences, 1D for the words and 1D for the word2vec dimensions.
    embedded_sentences = np.stack(
        [np.stack([embedding_weights[token] for token in sentence]) for sentence in padded_sequences])

    # Add back the saliency scores
    # input_output = np.column_stack((embedded_sentences,data[:,2]))

    input_output = np.array([])
    for i in range(len(data)):
        input_output = np.append(input_output, np.array([embedded_sentences[i], data[i, 2]]))

    del embedding_model

    return input_output


def rand_embed_sentences(data, NUM_WORDS=None):
    # setences ex: ["It's the first sentence!","it is the second sentence"]
    sentences = data[:, 1]
    tokenizer = Tokenizer(num_words=NUM_WORDS)

    # Tokenize the sentences, that is to say convert the 2 sentences ["It's the first sentence!","It is the second sentence"] to 2 sequences [[1 4 2 5 3],[1 6 2 7 3]]
    # It handles the ennoying cases (punctuation, Upper cases, etc...)
    tokenizer.fit_on_texts(sentences)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences)

    # print('padded_sequences shape:', padded_sequences.shape)

    return padded_sequences, data[:, 2]


if __name__ == "__main__":
    # For debugging purpose
    rand_embedded_sentences = rand_embed_sentences(np.array([[1, "hello!", 0.2],
                                                             [2, "cheese cake", 0.8]]))
    print(rand_embedded_sentences)
