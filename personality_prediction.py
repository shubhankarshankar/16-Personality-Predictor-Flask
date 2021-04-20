from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify, render_template, url_for, redirect
import pickle as pkl
import numpy as np
import pandas as pd
import tensorflow as tf
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

vocab_size = 10000
maxlen = 1500
trunc_type = "post"
pad_type = "post"
oov_tok = "<OOV>"
embedding_dim = 256
modelIE = load_model("models/model_IE.h5")
modelNS = load_model("models/model_NS.h5")
modelFT = load_model("models/model_FT.h5")
modelJP = load_model("models/model_JP.h5")

lemmatiser = WordNetLemmatizer()
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

# Remove the stop words for speed
useless_words = stopwords.words("english")

# Remove these from the posts
unique_type_list = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP',
                    'ENFP', 'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
unique_type_list = [x.lower() for x in unique_type_list]

# Splitting the MBTI personality into 4 letters and binarizing it

b_Pers = {'I': 0, 'E': 1, 'N': 0, 'S': 1, 'F': 0, 'T': 1, 'J': 0, 'P': 1}
b_Pers_list = [{0: 'I', 1: 'E'}, {0: 'N', 1: 'S'},
               {0: 'F', 1: 'T'}, {0: 'J', 1: 'P'}]


def translate_personality(personality):
    # transform mbti to binary vector
    return [b_Pers[l] for l in personality]

# To show result output for personality prediction


def translate_back(personality):
    # transform binary vector to mbti personality
    s = ""
    for i, l in enumerate(personality):
        s += b_Pers_list[i][l]
    return s


def pre_process_text(df, remove_stop_words=True, remove_mbti_profiles=True):
    list_personality = []
    list_posts = []
    len_df = len(df)
    i = 0

    for row in df.iterrows():
        # Remove and clean comments
        posts = row[1].posts

        # Remove url links
        temp = re.sub(
            'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', posts)

        # Remove Non-words - keep only words
        temp = re.sub("[^a-zA-Z]", " ", temp)

        # Remove spaces > 1
        temp = re.sub(' +', ' ', temp).lower()

        # Remove stop words
        if remove_stop_words:
            temp = " ".join([lemmatiser.lemmatize(w)
                             for w in temp.split(' ') if w not in useless_words])
        else:
            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])

        # Remove MBTI personality words from posts
        if remove_mbti_profiles:
            for t in unique_type_list:
                temp = temp.replace(t, "")

        # transform mbti to binary vector
        # or use lab_encoder.transform([row[1].type])[0]
        type_labelized = translate_personality(row[1].type)
        list_personality.append(type_labelized)
        # the cleaned df temp is passed here
        list_posts.append(temp)

    # returns the result
    list_posts = np.array(list_posts)
    list_personality = np.array(list_personality)
    return list_posts, list_personality


def predictIE(prediction):
    p = round(prediction[0][0])
    if p == 0:
        return 'I'
    else:
        return 'E'


def predictNS(prediction):
    p = round(prediction[0][0])
    if p == 0:
        return 'N'
    else:
        return 'S'


def predictFT(prediction):
    p = round(prediction[0][0])
    if p == 0:
        return 'F'
    else:
        return 'T'


def predictJP(prediction):
    p = round(prediction[0][0])
    if p == 0:
        return 'J'
    else:
        return 'P'


def presonality_prediction(post):
    sample_data = pd.DataFrame(data={'type': "INTJ", 'posts': [post]})
    clean_sample, _ = pre_process_text(sample_data)

    tokenizer.fit_on_texts(clean_sample)
    sample_sequence = tokenizer.texts_to_sequences(clean_sample)
    sample_sequence_padded = pad_sequences(
        sample_sequence, maxlen=maxlen, truncating=trunc_type, padding=pad_type)

    final_pred = predictIE(modelIE.predict(sample_sequence_padded)) + predictNS(modelNS.predict(sample_sequence_padded)) + \
        predictFT(modelFT.predict(sample_sequence_padded)) + \
        predictJP(modelJP.predict(sample_sequence_padded))

    return final_pred
