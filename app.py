from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, render_template, url_for, redirect
import pickle as pkl
import numpy as np
import pandas as pd
import re
import nltk
import personality_prediction

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World'


@app.route('/predict_personality/<string:post>', methods=['GET'])
def predict_personality(post):
    s = personality_prediction.presonality_prediction(post)
    return s, 201


if __name__ == "__main__":
    app.run(use_reloader=True)
