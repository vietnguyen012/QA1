from googleapiclient.discovery import build
import nltk
from numpy import random
import requests
from bs4 import BeautifulSoup
import timeout_decorator
from nltk import sent_tokenize
from multiprocessing import Pool
import re
import sys
import urllib3
from urllib3.exceptions import InsecureRequestWarning
urllib3.disable_warnings(category=InsecureRequestWarning)

api_key = ['AIzaSyD0sNrnah2WHJ7jtp5SFhIZxBpMYxaNndo',
            'AIzaSyA0OtrmUCghVL1z6OwMrt9xznRZDCWPldM',
            'AIzaSyDn2d6vH23lCK1sUlXACLCFDXIsyv7PUdk',
            'AIzaSyC0EjluoHln5U25h_aLMG-OoH17WQ0g4do',
            'AIzaSyDpkzUV2ONSbgVuVSgqoYb8FcdipFWiOnw',
            'AIzaSyBO_r7dnf8DyTqKPTJAKCmtsf-kuMI9IeM',
            'AIzaSyCMgFA7GV-xleVI_k0-YvwC-bLYMf21UN4',
            'AIzaSyA01lgW_qLBC0EzuBNausut12jZkPbnnBY',
            'AIzaSyA8KOP4NlQrtJLAjoa3lPf5-MuIbnjXZC4']
# api_key = ['AIzaSyDMkerl0FThx2K6EpFwPwByZHTs0t5YWe8']

Custom_Search_Engine_ID = "039db513229c6de10"

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

# @timeout_decorator.timeout(100)
def ggsearch(para):
    # try:
        i = para[0]
        service = para[1]
        query = para[2]
        if (i == 0):
            res = service.cse().list(q=query,cx = Custom_Search_Engine_ID, gl ='vn',
                                     googlehost = 'vn', hl = 'vi').execute()
        else:
            res = service.cse().list(q=query,cx = Custom_Search_Engine_ID,num=10,start = i*10, gl ='vn',
                                     googlehost = 'vn', hl = 'vi').execute()

        # print(res)
        # print('******************************************')
        # print(res[u'items'])
        return res[u'items']
    # except:
    #     print('Exception ggsearch')
    #     return []

@timeout_decorator.timeout(100)
def getContent(url):
    #get html file from url
    response = requests.get(url, verify=False, timeout = 20)
    soup = BeautifulSoup(response.content, "html.parser")

    #prepocessing
    try:
        soup.a.decompose()
    except:
        pass
    try:
        soup.script.decompose()
    except:
        pass
    try:
        soup.style.decompose()
    except:
        pass

    LIMIT = 300

    all_li_ele = soup.find_all(['li', 'p'])
    page_contents = []
    content = ''
    for e in all_li_ele:
        txt = e.get_text()
        txt_arr = txt.split('\n')
        txt_arr = [t.strip() for t in txt_arr if len(t.strip()) > 0]
        if not str(e).startswith('<li>'):
            for t in txt_arr:
                if len(content + t) <= LIMIT:
                    content += t + ' '
                else:
                    page_contents.append(content)
                    content = t
            content += '\n'
        else:
            content += ' '.join(txt_arr)
            content += '\n'

    for i in range(len(page_contents)):
        sentences = nltk.sent_tokenize(page_contents[i])
        sentences = [s for s in sentences if len(s) > 10 and '?' not in s]
        page_contents[i] = ' '.join(sentences)

    return '\n\n'.join(page_contents)
    # try:
        # html = requests.get(url, verify=False, timeout = 10)
        # tree = BeautifulSoup(html.text,'lxml')
        # for invisible_elem in tree.find_all(['script', 'style']):
        #     invisible_elem.extract()

        # paragraphs = [p.get_text() for p in tree.find_all("p")]

        # for para in tree.find_all('p'):
        #     para.extract()

        # for href in tree.find_all(['a','strong']):
        #     href.unwrap()

        # tree = BeautifulSoup(str(tree.html),'lxml')

        # # print(str(tree.html))

        # text = tree.get_text(separator='\n\n')
        # text = re.sub('\n +\n','\n\n',text)

        # paragraphs += text.split('\n\n')
        # paragraphs = [re.sub(' +',' ',p.strip()) for p in paragraphs]
        # paragraphs = [p for p in paragraphs if len(p.split()) > 10]

        # for i in range(0,len(paragraphs)):
        #     sents = []
        #     text_chunks = list(chunks(paragraphs[i],100000))
        #     for chunk in text_chunks:
        #         sents += sent_tokenize(chunk)

        #     sents = [s for s in sents if len(s) > 2]
        #     sents = ' . '.join(sents)
        #     paragraphs[i] = re.sub('\.\s*\.','.',sents)

        # return '\n\n'.join(paragraphs)
    # except:
    #     print('Cannot read ' + url)
    #     # print('Cannot read ' + url, str(sys.exc_info()[0]))
    #     return ''


class GoogleSearch():
    __instance = None

    def __init__(self):
        self.pool = Pool(4)

    def search(self,question):
        service = build("customSearch", "v1", developerKey=api_key[int(random.randint(0,len(api_key)-1))])
        # service = build("customSearch", "v1", developerKey=api_key[0])
        pages_content = ggsearch([0, service, question])
        pages_content = pages_content[:1]

        document_urls = set([])
        for page in pages_content:
            if 'fileFormat' in page:
                continue
            document_urls.add(page[u'link'])
        document_urls = list(document_urls)

        # document_urls = set(['https://vietnamese.cdc.gov/coronavirus/2019-ncov/symptoms-testing/symptoms.html'])

        gg_documents = self.pool.map(getContent,document_urls)
        gg_documents = [d for d in gg_documents if len(d) > 20]

        return document_urls,gg_documents
