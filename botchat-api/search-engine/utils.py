import json
import numpy as np

class json_utils:
    docs = []
    def __init__(self, json_paths = []):
        # try:
        for path in json_paths:
            with open(path) as f:
                data = json.load(f)
                self.docs.append(data)
        # except:
        #     raise Exception("Can't get data")


    def extract_json_context(self):
        contexts = []
        for doc in self.docs:
            for d in doc['data']:
                for para in d['paragraphs']:
                    context = np.char.replace(para['context'],'\\','')
                    contexts.append(str(context))
        return contexts

class passages_utils:
    def __init__(self, docs=[]):
        self.docs = docs
    
    def generate_passages(self, n = 6):
        passages = []
        
        for doc in self.docs:
            sentences = doc.rsplit('.')
            
            if len(sentences) <= n:
                passages.append(' '.join(sentences))
            else:
                for i in range(0,len(sentences) - n + 1):
                    passages.append(' '.join([sentences[i + j] for j in range(0,n) if '?' not in sentences[i + j]]))      
        return passages
    
    def extract_paragraphs(self):
        all_paragraphs = []
        for doc in self.docs:
            paragraphs = doc.split('\n\n')
            for para in paragraphs:
                # all_paragraphs.append(para)
                sentences = para.rsplit('.')
                sentences = [s for s in sentences if '?' not in s]

                if len(sentences) > 2:
                    all_paragraphs.append(para)
                # if len(sentences) <= 3:
                #     all_paragraphs.append(' '.join(sentences))
                # else:
                #     for i in range(0,len(sentences) - 3 + 1):
                #         all_paragraphs.append(' '.join([sentences[i + j] for j in range(0,3) if '?' not in sentences[i + j]]))
        return all_paragraphs