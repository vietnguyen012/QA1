import numpy as np
from nltk.tokenize import word_tokenize
from num2words import num2words
import string
import codecs

stopwords_path = '/home/vietnguyen/albert_vi/botchat-api/search-engine/resources/vietnamese-stopwords.txt' 
stopwords= set(w.strip() for w in codecs.open(stopwords_path, 'r',encoding='utf-8').readlines())
# stopwords = open('resources/vietnamese-stopwords.txt').read().split('\n')
punct_set = list(set([c for c in string.punctuation]) | set(['“','”',"...","–","…","..","•",'“','”','-']))
    
def convert_lower_case(doc):
    return np.char.lower(doc)

def remove_stopwords(doc):
    doc = word_tokenize(doc)
    doc = [word for word in doc if word not in set(stopwords)]
    doc = [word for word in doc if word.isalnum()]
    return doc

def remove_punctation(doc):
    for i in range (len(punct_set)):
        doc = np.char.replace(doc, punct_set[i],'')
        doc = np.char.replace(doc, " ", " ")
    return doc

def convert_numbers(doc):
    tokens = word_tokenize(str(doc))
    new_text = ""
    for token in tokens:
        try:
            token = num2words(int(token), lang='vi')
        except:
            pass
        new_text = new_text + " " + token
    return new_text.strip()

def preprocess_data(doc): 
    doc = convert_lower_case(doc)
    # doc = self.convert_numbers(doc)
    doc = remove_stopwords(str(doc))
    doc = remove_punctation(doc)
    return doc

def preprocess_query(query):
    query = query.lower()
    corona_synonyms = ['covid', 'corona', 'covid-19', 'covid 19','ncov','2019-ncov']
    
    isCoronaQuery = False
    for word in corona_synonyms:
        if word in query:
            isCoronaQuery = True
    if isCoronaQuery:
        for word in corona_synonyms:
            if word not in query:
                query += ' ' + word

    query = np.char.replace(query, '?', ' ')
    query = str(query).strip()
    query = convert_lower_case(query)
    # doc = self.convert_numbers(doc)
    query = remove_stopwords(str(query))
    query = remove_punctation(query)
    return query
