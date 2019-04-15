from flask import Flask,render_template,request
import os
import pyLDAvis.gensim
import spacy
spacy.load('en')
from spacy.lang.en import English
import nltk
import random
from gensim import corpora
import pickle
import gensim


parser = English()

def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


nltk.download('wordnet')

from nltk.corpus import wordnet as wn


def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma


from nltk.stem.wordnet import WordNetLemmatizer


def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)

nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))

def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens



app=Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def main():
    return render_template("index.html")
#
# @app.route('/lda')
# def showSignUp():
#     return render_template('lda.html')

@app.route('/viewResults')
def viewResults():
    return render_template('ldaHi.html')


@app.route('/showSignUp')
def showSignUp():
    return render_template('signup.html')


@app.route('/upload',methods=['POST'])
def upload():

    target = os.path.join(APP_ROOT,"upload/")


    if not os.path.isdir(target):
        os.mkdir(target)

    file = request.files['file']
    if file:

        destination="/".join([target,file.filename])
        file.save(destination)

    text_data = []
    with open(os.path.join(APP_ROOT,"upload" ,'dataset.csv')) as f:
        for line in f:
            tokens = prepare_text_for_lda(line)
            if random.random() > .99:
                print(tokens)
                text_data.append(tokens)

    dictionary = corpora.Dictionary(text_data)

    corpus = [dictionary.doc2bow(text) for text in text_data]

    pickle.dump(corpus, open('corpus.pkl', 'wb'))
    dictionary.save('dictionary.gensim')

    NUM_TOPICS = 5
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=NUM_TOPICS, id2word=dictionary, passes=15)
    ldamodel.save('model5.gensim')

    dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
    corpus = pickle.load(open('corpus.pkl', 'rb'))
    lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')

    lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
    pyLDAvis.save_html(lda_display,os.path.join(APP_ROOT,"templates/" ,'ldaHi.html'))
    # pyLDAvis.display(lda_display)

    return render_template("ldaHi.html")




if __name__=="__main__":
    app.run(debug=True)

