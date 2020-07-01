import numpy as np
import faiss

def find_idx_vector(query_title, arr_title, arr_vector):
    # find index
    idx = np.isin(arr_title, query_title)
    query_idx = np.where(idx)[0][0]

    # query vector (cleaned_title이 vector가 된 곳에서 query index로 찾기)
    # 실제 서비스에서 db로 대체 가능
    query_vector = arr_vector[query_idx]

    return query_idx, query_vector

## cleaned_arr_title 필요 없음.
def find_sim_title(query_vector, index, arr_title, K):
    scores = []
    sim_titles = []

    arr_sim_score, arr_sim_index = index.search(query_vector, K + 1)

    # print('\nTarget title : {0}'.format(arr_title[title_idx]))
    # print('Target cleaned title : {0}'.format(cleaned_arr_title[title_idx]))
    # print('-' * 50)

    # 자기자신은 제외 (정제 뒤 nan title의 경우 자기 자신 title 찾을 수 없음)
    # try:
    #     find_idx_self = np.where(arr_sim_index == title_idx)[0].item()
    #     arr_sim_index = np.delete(arr_sim_index, find_idx_self)
    #     arr_sim_score = np.delete(arr_sim_score, find_idx_self)
    #
    # except ValueError:
    #     pass

    num = 0

    arr_sim_index = arr_sim_index.reshape(-1)
    arr_sim_score = arr_sim_score.reshape(-1)

    for idx, score in zip(arr_sim_index, arr_sim_score):
        sim_titles.append(arr_title[idx])
        scores.append(score)
        # print(arr_title[idx])
        # print(score)

        if num > K-2:
            break
        num += 1

    # for sim_title in sim_titles:
    #     print(sim_title)
    #
    # for score in scores:
    #     print(score)
    #
    # print('\n')
    # print('=' * 50)

    return list(arr_sim_index), sim_titles, scores

def search_index(query_title, arr_title, index, arr_vector, vector_size, K):
    # title index, vector
    find_idx, find_vector = find_idx_vector(query_title, arr_title, arr_vector)

    query_vector = find_vector.reshape(-1, vector_size)
    sim_titles, scores = find_sim_title(query_vector, index, arr_title, K)

    return sim_titles, scores

def find_range_sim_title(query_vector, index, arr_title, threshold):
    scores = []
    sim_titles = []

    _, arr_sim_score, arr_sim_index = index.range_search(query_vector, threshold)

    # print('\nTarget title : {0}'.format(arr_title[title_idx]))
    # print('Target cleaned title : {0}'.format(cleaned_arr_title[title_idx]))
    # print('-' * 50)

    # 자기자신은 제외 (정제 뒤 nan title의 경우 자기 자신 title 찾을 수 없음)
    # try:
    #     find_idx_self = np.where(arr_sim_index == title_idx)[0].item()
    #     arr_sim_index = np.delete(arr_sim_index, find_idx_self)
    #     arr_sim_score = np.delete(arr_sim_score, find_idx_self)
    #
    # except ValueError:
    #     pass

    arr_sim_index = arr_sim_index.reshape(-1)
    arr_sim_score = arr_sim_score.reshape(-1)

    # 실제 서비스에서는 db로 교체
    for idx, score in zip(arr_sim_index, arr_sim_score):
        sim_titles.append(arr_title[idx])
        scores.append(score)

    # for sim_title in sim_titles:
    #     print(sim_title)
    #
    # for score in scores:
    #     print(score)
    #
    # print('\n')
    # print('=' * 50)

    return arr_sim_index, sim_titles, scores

def _search_range_index(query_title, arr_title, index, arr_vector, threshold):

    vector_dim = arr_vector[0].shape[0]

    # title index, vector
    query_idx, query_vector = self._find_idx_vector(query_title, arr_title, arr_vector)

    query_vector = query_vector.reshape(-1, vector_dim)
    arr_sim_index, sim_titles, scores = self._find_range_sim_query_title(query_vector, index, arr_title, threshold)

    return arr_sim_index, sim_titles, scores

def find_intersection(query_title, arr_title, img_index, title_index, img_vector, query_vector, img_thrshold = 0.1, title_threshold = 0.1):

    # 1차 img
    img_sim_idx, img_sim_titles, img_scores = search_range_index(query_title, arr_title, img_index, img_vector, threshold = img_thrshold)

    # 2차 title
    title_sim_idx, title_sim_titles, title_scores = search_range_index(query_title, arr_title, title_index, query_vector, threshold = title_threshold)

    intersect_idx = np.intersect1d(img_sim_idx, title_sim_idx)

    get = arr_title[intersect_idx]

    return intersect_idx, get

# code example
# search_range_index(title,arr_title,cleaned_arr_title,index,minmax_word_vectors,200,0.2)
