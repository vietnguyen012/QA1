import re
import unicodedata

def convert_lower_case(doc):
    return doc.lower()

def remove_control_char(doc):
    chars = [char for char in doc]
    for i in range(len(chars)):
        if unicodedata.category(chars[i])[0] == "C":
            chars[i] = " "
    return "".join(chars)

def remove_non_word_char(doc):
    return "".join(ch for ch in doc if ch.isalnum() or ch in [',','.',' ','-'])

def remove_redundant_space(doc):
    doc = re.sub('\\s+',' ', doc)
    return doc.strip()

def add_dot(doc):
    doc = doc.strip()
    if doc[0] != '.':
        doc = '.' + doc
    if doc[len(doc)-1] != '.':
        doc += '.'
    return doc

def post_process(docs):
    res = []
    for doc in docs:
        doc = convert_lower_case(doc)
        doc = remove_control_char(doc)
        doc = remove_non_word_char(doc)
        doc = remove_redundant_space(doc)
        doc = add_dot(doc)
        res.append(doc)
    return res