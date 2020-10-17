from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec,KeyedVectors
from nltk import word_tokenize
from pyemd import emd
import pandas as pd
import numpy as np
import codecs
import re
import time
import preprocess

def LogInfo(stri):
    print(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+'  '+stri)
    
def preprocess_data_vi(stopwords,doc):
    '''
    Function: preprocess data in Chinese including cleaning, tokenzing...
    Input: document string
    Output: list of words
    '''     
    doc = doc.lower()
    doc = word_tokenize(doc)
    doc = [word for word in doc if word not in set(stopwords)]
    doc = [word for word in doc if word.isalnum()]
    return doc

def filter_words(vocab,doc):
    '''
    Function: filter words which are not contained in the vocab
    Input:
        vocab: list of words that have word2vec representation
        doc: list of words in a document
    Output:
        list of filtered words
    '''
    return [word for word in doc if word in vocab]

def f(x):
    if x<0.0: return 0.0
    else: return x
    
def handle_sim(x):  
    return 1.0-np.vectorize(f)(x)

def regularize_sim(sims):
    '''
    Function: replace illegal similarity value -1 with mean value
    Input: list of similarity of document pairs
    Output: regularized list of similarity 
    '''
    sim_mean = np.mean([sim for sim in sims if sim!=-1])
    r_sims = []
    errors = 0
    for sim in sims:
        if sim==-1:
            r_sims.append(sim_mean)
            errors += 1
        else:
            r_sims.append(sim)
#     LogInfo('Regularize: '+str(errors))
    return r_sims

def load_word2vec(model_path):
    model = dict()
    for line in open(model_path,encoding='utf-8'):
        l = line.strip().split()    
        st=' '.join(l[:-300]).lower()   
        model[st]=list(map(float,l[-300:]))
  
    # num_keys=len(model)
   
    return model


def wmd_sim(lang,docs1,docs2):
    '''
    Function:
        calculate similarity of document pairs 
    Input: 
        lang: text language-Chinese for 'cn'/ English for 'en'
        docs1:  document strings list1
        docs2: document strings list2
    Output:
        similarity list of docs1 and docs2 pairs: value ranges from 0 to 1; 
                  
    '''
    # check if the number of documents matched
    assert len(docs1)!=0,'Documents list1 is null'
    assert len(docs2)!=0,'Documents list2 is null'
    assert lang=='vi', 'Language setting is wrong'
    # change setting according to text language 
    if lang=='vi':
        model_path = '/home/vietnguyen/albert_vi/botchat-api/search-engine/resources/w2v_model_ug_cbow.word2vec'
        # stopwords_path = 'resources/vietnamese-stopwords.txt'
        # preprocess_data = preprocess_data_vi
    # load word2vec model  
    LogInfo('Load word2vec model...')
#     model = load_word2vec('../model/sgns.baidubaike.bigram-char')
#     vocab = list(model.keys())
    model = KeyedVectors.load(model_path)
    # model = KeyedVectors.load_word2vec_format(model_path,binary=True, encoding='utf-8',unicode_errors='ignore')
    vocab = model.wv.vocab

    # preprocess data
    # stopwords= set(w.strip() for w in codecs.open(stopwords_path, 'r',encoding='utf-8').readlines())
    sims = []
    LogInfo('Calculating similarity...')
    for i in range(len(docs1)):        
        p1 = preprocess.preprocess_query(docs1[i])
        p2 = preprocess.preprocess_data(docs2[i])
        p1 = filter_words(vocab,p1)
        p2 = filter_words(vocab,p2)
        if len(p1)==0 or len(p2)==0:
            # if any filtered document is null, return -1 
            sim = -1
        else:
            p1 = ' '.join(p1)
            p2 = ' '.join(p2)
            vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w+\b', stop_words=None)
            v1,v2 = vectorizer.fit_transform([p1,p2])
            # pyemd needs double precision input
            v1 = v1.toarray().ravel().astype(np.double)
            v2 = v2.toarray().ravel().astype(np.double)
            # transform word count to frequency [0,1]
            v1 /= v1.sum()
            v2 /= v2.sum()
            # obtain word2vec representations 
            W = [model[word] for word in vectorizer.get_feature_names()]
            # calculate distance matrix (distance = 1-cosine similarity) [0,1]
            D = handle_sim(cosine_similarity(W)).astype(np.double)         
            # calculate minimal distance using EMD algorithm
            min_distance = emd(v1,v2,D)
            # calculate similarity (similarity = 1-min_distance)
            sim = 1-min_distance
        
        sims.append(sim)
    # regularize similarity: replace -1 with average similarity
    rsims = regularize_sim(sims) 
    # 只保留小数点后四位
    rsims = [round(sim,4) for sim in rsims]
    return rsims

def compute_ser(sims):
    '''
    Function: compute SER(semantic error rate) according to the document similarity
    Input: 
        sims: list of document similarity
    Output:
        sers: list of document SER
    '''
    sers = [round(1.0-sim,4) for sim in sims]
    return sers

def example():
    # English text example
    docs1 = ['Corona là gì',
                 'Corona 19 là gì',
                'Corona 19 là gì']
    docs2 = ['Trong kỳ nghỉ Tết Nguyên đán vừa qua, khi Việt Nam phát hiện hai bệnh nhân đầu tiên nhiễm virus Corona (2019-nCoV) thì cơn “bão” thông tin bắt đầu bùng lên mạnh mẽ khiến nhiều người chưa hiểu rõ về bệnh, dịch lo lắng, sợ hãi, thậm chí hoảng loạn. Virus Corona gây bệnh viêm phổi cấp ở Vũ Hán là gì? . Vì sao virus Corona lại lan nhanh thành dịch chỉ trong thời gian ngắn? . Phòng ngừa virus Corona bằng cách nào?… là những thắc mắc cần được thông tin chuẩn xác.',
                'Các nhà khoa học Trung Quốc cho biết, trung bình một bệnh nhân nhiễm virus Corona sẽ lây lan sang 5,5 người khác. Chính vì virus Corona có khả năng lan truyền rất nhanh từ người sang người, nên nếu người dân không được trang bị kiến thức về phòng chống bệnh, đại dịch rất dễ xảy ra.',
                'Virus Corona là một loại virus gây ra tình trạng nhiễm trùng trong mũi, xoang hoặc cổ họng. Có 7 loại virus Corona, trong đó, 4 loại không nguy hiểm là 229E, NL63, OC43 và HKU1; hai loại khác là MERS-CoV và SARS-CoV, nguy hiểm hơn và từng gây ra đại dịch toàn cầu. Bên cạnh đó, còn một loại virus Corona thuộc chủng mới (ký hiệu 2019-nCoV hoặc nCoV, còn được gọi với cái tên “Virus Vũ Hán”) đang “tung hoành” trong những ngày này. Đây là tác nhân gây ra bệnh viêm phổi cấp, khiến hàng nghìn người nhiễm bệnh và làm số ca tử vong không ngừng tăng lên từng ngày.']
    # calculate similarity
    sims = wmd_sim('vi',docs1,docs2)
    # calculate SER
    sers = compute_ser(sims)
    # print result
    for i in range(len(sims)):
        print(docs1[i])
        print(docs2[i])
        print('Similarity: %.4f' %sims[i])
        print('SER: %.4f' %sers[i])
        
def relevance_ranking_vi(docs1, docs2):
    sims = wmd_sim('vi',docs1,docs2)
    sers = compute_ser(sims)

    return sims, sers

if __name__ == '__main__':
    example()
 
