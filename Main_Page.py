import streamlit as st
import pandas as pd
import nltk
import numpy as np
import re
import ast
from PIL import Image
import plotly_express as px
from sklearn import svm
from nltk.corpus import stopwords
from deep_translator import GoogleTranslator
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

"""
# Sentiment Analysis App
This App is using The Mixed of Machine Learning and Lexicon Based Learning to do Sentiment Analyze.
The Algorithms used Are Multinomial Naive Bayes, Support Vector Machine, and Lexicon Based.
"""

st.sidebar.markdown("# Main page")

image = Image.open('emotions.jpeg')
st.image(image, caption="Sentiment Analysis describe the user Emotions or Satisfaction about Something. Let's try analyze sentiment!")

# read a CSV file inside the 'data" folder next to 'app.py'
df = pd.read_csv("./cleaned_data_translated.csv")

analyser = SentimentIntensityAnalyzer()

scores = [analyser.polarity_scores(x)
          for x in df['text_translated']]

df['compound_score'] = [x['compound'] for x in scores]

conditions = [
    (df['compound_score'] > 0), (df['compound_score']
                                 == 0), (df['compound_score'] < 0)
]
choices = ["positive", "neutral", "negative"]
df['klasifikasi'] = np.select(
    conditions, choices, default='neutral')

# shuffle data
d_shuffle = df.sample(frac=1)
# memetakan atribut dan label
vectorizer = CountVectorizer()
# banyak data yang akan ditraining
percent_training = int(len(df)*0.75)
d_train = d_shuffle.iloc[:percent_training]
d_test = d_shuffle.iloc[percent_training:]

# data train 70:30
df_train_att = vectorizer.fit_transform(d_train['liststring'])
df_test_att = vectorizer.transform(d_test['liststring'])

# data test
df_train_label = d_train['klasifikasi']
df_test_label = d_test['klasifikasi']

nb = MultinomialNB()
# fit data train 75:25
clf = nb.fit(df_train_att, df_train_label)

y_pred = clf.predict(df_test_att)

# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
classifier_svm = classifier_linear.fit(df_train_att, df_train_label)

y_pred2 = classifier_svm.predict(df_test_att)

d_test['nb_predicted'] = y_pred
d_test['svm_predicted'] = y_pred2

list_data = list(zip(d_test['liststring'], d_test['compound_score'],
                     d_test['nb_predicted'], d_test['svm_predicted']))

# st.subheader("Classify Result Based on The Different Algorithm")
# st.write(d_test)
# st.subheader("Classify Result Based on The Different Algorithm In List Format")
# st.write(list_data)


option = st.sidebar.selectbox(
    'Insert Text or Upload Files',
    ('Insert Text', 'Upload File'))

st.sidebar.write('You selected:', option)


def remove_pattern(text, pattern_regex):
    r = re.findall(pattern_regex, text)
    for i in r:
        text = re.sub(i, '', text)
    return text


def remove_symbol(text):
    text = ' '.join(
        re.sub("(@[A-Za-z0-9]+) | ([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())
    return text


# Remove Puncutuation
clean_spcl = re.compile('[/(){}\[\]\|@,;]')
clean_symbol = re.compile('[^0-9a-zA-Z]')


def clean_punct(text):
    text = clean_spcl.sub(' ', text)
    text = clean_symbol.sub(' ', text)
    return text


def _normalize_whitespace(text):
    corrected = str(text)
    corrected = re.sub(r"//t", r"\t", corrected)
    corrected = re.sub(r"( )\1+", r"\1", corrected)
    corrected = re.sub(r"(\n)\1+", r"\1", corrected)
    corrected = re.sub(r"(\r)\1+", r"\1", corrected)
    corrected = re.sub(r"(\t)\1+", r"\1", corrected)
    return corrected.strip(" ")


# add stopwords
nltk.download('stopwords')
list_stopwords = list(stopwords.words('indonesian'))
factory = StopWordRemoverFactory()
more_stopword = ['yg', 'aja', 'nya', 'ya', 'ga', 'iso',
                 'dr', 'trims', 'mah', 'gk', 'gak', 'sdh', 'skrg', 'trs']
more_list_stopwords = open('stopword_list_tala.txt').read()
more_list_stopwords = more_list_stopwords.split()
data_stopwords = list_stopwords+more_stopword+factory.get_stop_words()
stopword = factory.create_stop_word_remover()

# remove stopword


def clean_stopwords(text):
    text = ' '.join(word for word in text.split()
                    if word not in data_stopwords)
    return text


# import Sastrawi package
# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# menghapus slang word

# importing the module

# reading the data from the file
# https://github.com/louisowen6/NLP_bahasa_resources
# A Curated List of Dataset and Usable Library Resources for NLP in Bahasa Indonesia
with open('combined_slang_words.txt') as f:
    data = f.read()
# reconstructing the data as a dictionary
slangs_new = ast.literal_eval(data)


def sentiment_calc(text):
    try:
        return text.split()
    except:
        return None

# do the slang remover function


def checkslang(text):
    # datalist = []
    for u, t in enumerate(text):
        if t in slangs_new.keys():
            text[u] = slangs_new[t]
    return print(' '.join(text))


def untokenized(text):
    if text != None:
        for a in text:
            data = print(' '.join([str(a)]))
        return data


if option == 'Insert Text':
    txt = st.sidebar.text_input('Text to analyze',)
    if len(txt) > 0:
        text_new = remove_symbol(txt)
        text_new = clean_punct(text_new)
        text_new = _normalize_whitespace(text_new)
        text_new = text_new.lower()
        text_new = clean_stopwords(text_new)
        text_new = stemmer.stem(text_new)
        text_new = text_new.split()
        # text_new = checkslang(text_new)
        for u, t in enumerate(text_new):
            if t in slangs_new.keys():
                text_new[u] = slangs_new[t]
        data_new = ' '.join(text_new)
        st.write('Cleaned text :', data_new)
        result_nb = clf.predict(vectorizer.transform([data_new]))
        st.write('Predicted Sentiment Using NB:', result_nb)
        st.write('Accuracy using NB:', clf.score(df_test_att, df_test_label))
        result_svm = classifier_svm.predict(vectorizer.transform([data_new]))
        st.write('Predicted Sentiment Using SVM:', result_svm)
        st.write('Accuracy using SVM:', classifier_svm.score(
            df_test_att, df_test_label))
else:
    uploaded_file = st.sidebar.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = df.sample(frac=1)
        st.header("Dataset Uploaded")
        st.write(df)
        st.header("Data Used")
        st.write(df['content'].head())
        df['removed_symbol'] = df['content'].apply(lambda x: remove_symbol(x))
        df['clean_punct'] = df['removed_symbol'].apply(clean_punct)
        df['clean_double_space'] = df['clean_punct'].apply(
            _normalize_whitespace)
        df['case_folding'] = df['clean_double_space'].apply(
            lambda x: " ".join(x.lower() for x in x.split()))
        df['clean_sw'] = df['case_folding'].apply(clean_stopwords)
        df['stemmed'] = [stemmer.stem(x) for x in df['clean_sw']]
        df['B'] = df['stemmed'].apply(sentiment_calc)
        df['B'].apply(checkslang)
        df['liststring'] = df['B'].apply(lambda x: ' '.join(map(str, x)))
        data_test = vectorizer.transform(df['liststring'])
        y_pred_nb = clf.predict(data_test)
        y_pred_svm = classifier_svm.predict(data_test)
        df['nb_predicted'] = y_pred_nb
        df['svm_predicted'] = y_pred_svm
        st.write('Predicted Sentiment Using NB:', y_pred_nb)
        st.write('Predicted Sentiment Using SVM:', y_pred_svm)
        st.header('Classification Result Using The Different Algorithm')
        list_data = list(zip(df['liststring'],
                             df['nb_predicted'], df['svm_predicted']))
        st.write(df[['content', 'liststring', 'nb_predicted', 'svm_predicted']])
        st.header("Positive Classification")
        if df.loc[df['nb_predicted'] == 'positive'] is not None:
            df['filter_pos_nb'] = (
                df['content'].loc[df['nb_predicted'] == 'positive'])
            st.write(df['filter_pos_nb'])
        if df.loc[df['svm_predicted'] == 'positive'] is not None:
            df['filter_pos_svm'] = (
                df['content'].loc[df['svm_predicted'] == 'positive'])
            st.write(df['filter_pos_svm'])
        st.header("Neutral Classification")
        if df.loc[df['nb_predicted'] == 'neutral'] is not None:
            df['filter_net_nb'] = (
                df['content'].loc[df['nb_predicted'] == 'neutral'])
            st.write(df['filter_net_nb'])
        if df.loc[df['svm_predicted'] == 'neutral'] is not None:
            df['filter_net_svm'] = (
                df['content'].loc[df['svm_predicted'] == 'neutral'])
            st.write(df['filter_net_svm'])
        st.header("Negative Classification")
        if df.loc[df['nb_predicted'] == 'negative'] is not None:
            df['filter_neg_nb'] = (
                df['content'].loc[df['nb_predicted'] == 'negative'])
            st.write(df['filter_neg_nb'])
        if df.loc[df['svm_predicted'] == 'negative'] is not None:
            df['filter_neg_svm'] = (
                df['content'].loc[df['svm_predicted'] == 'negative'])
            st.write(df['filter_neg_svm'])
