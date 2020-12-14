from flask import Flask, session, redirect, url_for, request, render_template
from flask_bootstrap import Bootstrap
import json
import numpy as np
import pandas as pd
import math
import os
from sklearn.metrics import mean_squared_error as mse
from sklearn.naive_bayes import MultinomialNB
from copy import deepcopy
import pickle
import os, re, string
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)

bootstrap = Bootstrap(app)

filename = 'nb_model_final.sav'
loaded_model = pickle.load(open(filename, 'rb'))
fname = './vocabulary.json' 
with open(fname, 'r') as f:
    vocabulary = json.load(f)

@app.route('/', methods=['GET', 'POST'])
def login():
    print('check1')
    if request.method == 'POST':
        print('check1')
        review = request.form['review']
        print('check1')
        rating = predict(preproc(review))
        print(rating)
        # return render_template('index.html', rating=rating)
        return """<form method="post">
                    <div class="row">
                        <div class="col">
                            <input type="text" class="form-control" name=review placeholder="Add review here">
                        </div>
                        <div class="col">
                            <input type=submit value="Predict Score" class="btn btn-primary mb-2"></button>
                        </div>
                    </div>
                </form>
                <div class="row">
                    <div class="col-md-2 col-lg-2 col-xl-2"></div>
                    <div class="col-md-8 col-lg-8 col-xl-8"><h3>Score: {}</h3></div>
                    
                </div>""".format(rating)
    return """<form method="post">
            <div class="row">
                <div class="col">
                    <input type="text" class="form-control" name=review placeholder="Add review here">
                </div>
                <div class="col">
                    <input type=submit value="Predict Score" class="btn btn-primary mb-2"></button>
                </div>
            </div>
        </form>"""

def preproc(incomingString):
    return incomingString.translate(str.maketrans('', '', string.punctuation)).lower()

def predict(text):
    count_vect = CountVectorizer(vocabulary=vocabulary)
    test = count_vect.fit_transform([text])
    return loaded_model.predict(test)[0]