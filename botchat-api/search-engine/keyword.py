from underthesea import pos_tag, word_tokenize
import preprocess
from nltk.tokenize import word_tokenize as word_tokenize1

while True:
    question = input('Query: ')
    print(pos_tag(question))
    print(word_tokenize(question))
    print(preprocess.remove_stopwords(question.lower()))
    print(word_tokenize1(question.lower()))

    # 1Q nC --> relevance 1Q-1C-Score --> top kC
    # 1Q -> VECTOR, 1C -> VECTOR
    # 1 ele vector = 1 word
    
