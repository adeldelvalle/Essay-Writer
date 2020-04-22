import numpy as np
import re
import nltk
import spacy
from spacy.lang.en import English
from nltk.tokenize import word_tokenize
from collections import Counter
from nltk.corpus import stopwords
import re, string, unicodedata
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk import sent_tokenize, word_tokenize
import pandas as pd
from nltk import word_tokenize, sent_tokenize
import requests
from bs4 import BeautifulSoup


class Page:
    def __init__(self, link):
        self.page = requests.get(link)
        self.content = BeautifulSoup(self.page.content, 'html.parser')
        self.text = self.content.get_text()
        self.clean_text = ""
        self.data = self.pre_processing()

    def pre_processing(self):
        self.clean_text = sent_tokenize(self.text)
        return pd.DataFrame(data=self.clean_text, columns=["clean_text"])


class Document:
    """ Retrieve the narratives from the DataFrame and respectively
        store and pre-process it.

        :param df: DataFrame including the reports and the predictor variable.

        :ivar data: Stores the DataFrame.
        :ivar text: Stores the narratives as string.
        :ivar corpus: Stores the pre-processed text.
    """

    def __init__(self, df):
        self.data = df
        self.text = df["clean_text"].astype(str)
        self.textPreProcessing()

    def remove_non_ascii(self, words):
        """Remove non-ASCII characters from list of tokenized words

        :param words:  List of words to be transformed when removing non_ascii characters.

        :return new_words: List of words after the transformation of removed non_ascii characters.

        """
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words

    def remove_punctuation(self, words):
        """Remove punctuation from list of tokenized words

        :param words:  List of words that will get remove their punctuations, if any.

        :return new_words: List of transformed words.
        """
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words

    def stem_words(self, words):
        """Stem words in list of tokenized words

        :param words:  List of words to be processed.

        :return new_words: List of the received words respective stems.


        """

        stemmer = LancasterStemmer()
        stems = []
        for word in words:
            stem = stemmer.stem(word)
            stems.append(stem)
        return stems

    def lemmatize_verbs(self, words):
        """Lemmatize verbs in list of tokenized words

        :param words:  List of words to be processed.

        :return new_words: List of the received words respective lemmas.

        """

        lemmatizer = WordNetLemmatizer()
        lemmas = []
        for word in words:
            lemma = lemmatizer.lemmatize(word, pos='v')
            lemmas.append(lemma)
        return lemmas

    def remove_stopwords(self, words):
        """Remove common words that have no meaning or importance in the sentence.

       :param words:  List of words to be processed and get stop words removed..

       :return new_words: List of words with the stop words already removed.

       """

        stop_words = set(stopwords.words('english'))

        for word in stop_words:
            if word in words:
                words.remove(word)

        return words


    def normalize(self, words):
        words = self.remove_non_ascii(words)
        words = self.remove_stopwords(words)
        words = self.remove_punctuation(words)
        words = self.lemmatize_verbs(words)
        return words


    def textPreProcessing(self):
        """Pre-process the text, normalize and clean it.
           The function stores the cleaned text in the self.data
           attribute. """

        clean_text = []

        for sentence in self.text:
            sentence = word_tokenize(sentence)
            sentence = self.normalize(sentence)
            clean_text.append(sentence)


        self.data["clean_text"] = clean_text


page = Page("https://es.wikipedia.org/wiki/Pandemia_de_enfermedad_por_coronavirus_de_2019-2020")
documento = Document(page.data)
print(page.data.head())
print(documento.data.head())
