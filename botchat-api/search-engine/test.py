import preprocess
import postprocess
import utils
from gg_search import GoogleSearch
ggsearch = GoogleSearch()
import wmd_similarity

def top_k_contexts(query, k):
    links, documents = ggsearch.search(query)

    for link in links:
        print(link)

    paragraph_generator = utils.passages_utils(docs=documents)

    paragraphs = paragraph_generator.extract_paragraphs()

    queries = [query] * len(paragraphs)

    sims, sers = wmd_similarity.relevance_ranking_vi(queries, paragraphs)
    print(query)
    res = dict(zip(paragraphs, sims))
    final_res = sorted(res.items(), key = lambda x:x[1], reverse=True)

    count = 0
    for r in final_res:
        if '111' in r[0]:
            print(r[0], r[1])
            print('\n')
        count += 1
        # if count >= k:
        #     break
         

    res_lst = (list(r[0] for r in final_res))

    # for re in res_lst:
    #     print(re)
    #     print('\n')

    return postprocess.post_process(res_lst[:min(k, len(res_lst))])

if __name__ == "__main__":
    res = top_k_contexts('Covid 19 đến từ đâu',4)
    # for re in res:
    #     print(re)
    #     print('\n')
    # print(type(preprocess.preprocess_data('Covid 19 là gì')))

