import numpy as np
import faiss

def find_idx_vector(find_title, title_array, word_vectors):
    # find index
    idx = np.isin(title_array, find_title)
    find_idx = np.where(idx)[0][0]

    # find vector
    # 실제 서비스에서 db로 대체 가능
    find_vector = word_vectors[find_idx]

    return find_idx, find_vector

## cleaned_title_array 필요 없음.
def find_sim_title(title_vector, index, title_array, K):
    scores = []
    sim_titles = []

    sim_score_array, sim_index_array = index.search(title_vector, K + 1)

    # print('\nTarget title : {0}'.format(title_array[title_idx]))
    # print('Target cleaned title : {0}'.format(cleaned_title_array[title_idx]))
    # print('-' * 50)

    # 자기자신은 제외 (정제 뒤 nan title의 경우 자기 자신 title 찾을 수 없음)
    # try:
    #     find_idx_self = np.where(sim_index_array == title_idx)[0].item()
    #     sim_index_array = np.delete(sim_index_array, find_idx_self)
    #     sim_score_array = np.delete(sim_score_array, find_idx_self)
    #
    # except ValueError:
    #     pass

    num = 0

    sim_index_array = sim_index_array.reshape(-1)
    sim_score_array = sim_score_array.reshape(-1)

    for idx, score in zip(sim_index_array, sim_score_array):
        sim_titles.append(title_array[idx])
        scores.append(score)
        # print(title_array[idx])
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

    return list(sim_index_array), sim_titles, scores

def search_index(find_title, title_array, index, wordvector, vector_size, K):
    # title index, vector
    find_idx, find_vector = find_idx_vector(find_title, title_array, wordvector)

    title_vector = find_vector.reshape(-1, vector_size)
    sim_titles, scores = find_sim_title(title_vector, index, title_array, K)

    return sim_titles, scores

def find_range_sim_title(title_vector, index, title_array, threshold):
    scores = []
    sim_titles = []

    _, sim_score_array, sim_index_array = index.range_search(title_vector, threshold)

    # print('\nTarget title : {0}'.format(title_array[title_idx]))
    # print('Target cleaned title : {0}'.format(cleaned_title_array[title_idx]))
    # print('-' * 50)

    # 자기자신은 제외 (정제 뒤 nan title의 경우 자기 자신 title 찾을 수 없음)
    # try:
    #     find_idx_self = np.where(sim_index_array == title_idx)[0].item()
    #     sim_index_array = np.delete(sim_index_array, find_idx_self)
    #     sim_score_array = np.delete(sim_score_array, find_idx_self)
    #
    # except ValueError:
    #     pass

    sim_index_array = sim_index_array.reshape(-1)
    sim_score_array = sim_score_array.reshape(-1)

    # 실제 서비스에서는 db로 교체
    for idx, score in zip(sim_index_array, sim_score_array):
        sim_titles.append(title_array[idx])
        scores.append(score)

    # for sim_title in sim_titles:
    #     print(sim_title)
    #
    # for score in scores:
    #     print(score)
    #
    # print('\n')
    # print('=' * 50)

    return sim_index_array, sim_titles, scores

def search_range_index(find_title, title_array, index, vector, threshold):

    vector_dim = vector[0].shape[0]

    # title index, vector
    find_idx, find_vector = find_idx_vector(find_title, title_array, vector)

    title_vector = find_vector.reshape(-1, vector_dim)
    sim_index_array, sim_titles, scores = find_range_sim_title(title_vector, index, title_array, threshold)

    return sim_index_array, sim_titles, scores

def find_intersection(find_title, title_array, img_index, title_index, img_vector, title_vector, img_thrshold = 0.1, title_threshold = 0.1):

    # 1차 img
    img_sim_idx, img_sim_titles, img_scores = search_range_index(find_title, title_array, img_index, img_vector, threshold = img_thrshold)

    # 2차 title
    title_sim_idx, title_sim_titles, title_scores = search_range_index(find_title, title_array, title_index, title_vector, threshold = title_threshold)

    intersect_idx = np.intersect1d(img_sim_idx, title_sim_idx)

    get = title_array[intersect_idx]

    return intersect_idx, get

# code example
# search_range_index(title,title_array,cleaned_title_array,index,minmax_word_vectors,200,0.2)
