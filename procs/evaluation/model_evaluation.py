from evaluation.title_search_using_faiss import *
import numpy as np
import copy

def evaluate_preprocessing(origin, get):
    tmp_origin = copy.copy(origin)
    tmp_get = copy.copy(get)

    for title in tmp_get:
        ori_idx = np.where(tmp_origin == title)
        get_idx = np.where(tmp_get == title)

        # 배열의 size check
        # 예측한 값이 정답에 포함되지 않을 경우 연산 x
        array_size = ori_idx[0].size

        if array_size:
            ori_idx = ori_idx[0][0]
            get_idx = get_idx[0][0]

            # 정답과 예측한 배열에서 찾은 값들을 하나씩 제거 (성능평가를 위해)
            tmp_origin = np.delete(tmp_origin,ori_idx)
            tmp_get = np.delete(tmp_get,get_idx)

    return tmp_origin, tmp_get

def evaluate_performance(get, origin, tmp_origin, tmp_get):
    # 1 : 정답 발견 확률 (모델이 예측한 정답 수 / 전체 정답 수)
    acc = (len(origin)-len(tmp_origin))/len(origin)

    # 2. : 정답이라고 맞춘 수에서 정답이었던 확률 (모델이 예측한 정답 수 / 모델 예측 수) = precision
    prec = (len(get)-len(tmp_get))/len(get)
    return acc, prec

def cal_cosine_similarity(find_vector,sim_title_vector):
    similarity_array = np.ones(len(sim_title_vector))

    for num, sim_vector in enumerate(sim_title_vector):
        similarity = np.inner(find_vector, sim_vector) / (np.linalg.norm(find_vector) * np.linalg.norm(sim_vector))
        similarity_array[num] = similarity

    return similarity_array

def find_answer(find_title, total_df):

    find_df = total_df[total_df['title'] == find_title]
    groupNo, groupCnt = find_df['group_no'].iloc[0], find_df['group_cnt'].iloc[0]

    # # 정답 제목
    origin = total_df[total_df['group_no'] == groupNo].title.values

    return origin

def weight_filter(find_weight, sim_weight_array, sim_idx):
    equal_weight_array = np.where(sim_weight_array == find_weight)[0]
    final_sim_array = sim_idx[equal_weight_array]
    return final_sim_array

# intersection
def evaluate_model(find_title, total_df, title_array, img_index, title_index, img_vector, title_vector, img_thrshold=0.1, title_threshold=0.1, weight_check=True):
    origin = find_answer(find_title, total_df)
    # 예측 array
    sim_index_array, get = find_intersection(find_title, title_array, img_index, title_index, img_vector, title_vector, img_thrshold, title_threshold)

    ## 2 필터링 (중량)
    if weight_check:
        find_weight = total_df[total_df['title'] == find_title].weight.iloc[0]
        if str(find_weight) == 'nan':
            weight_check = False

    if weight_check:
        sim_weight_array = total_df.weight[sim_index_array]
        sim_index_array = weight_filter(find_weight,sim_weight_array,sim_index_array)

    get = title_array[sim_index_array]
    tmp_origin, tmp_get = evaluate_preprocessing(origin, get)
    acc, prec = evaluate_performance(get, origin, tmp_origin, tmp_get)

    return acc, prec

def title_evaluate_model(find_title, total_df, title_array, title_index, title_vector, title_threshold=0.1, weight_check=True):

    origin = find_answer(find_title, total_df)

    # 예측 array
    title_sim_idx, title_sim_titles, title_scores = search_range_index(find_title, title_array, title_index, title_vector, title_threshold)

    ## 2 필터링 (중량)
    if weight_check:
        find_weight = total_df[total_df['title'] == find_title].weight.iloc[0]
        if str(find_weight) == 'nan':
            weight_check = False

    if weight_check:
        sim_weight_array = total_df.weight[title_sim_idx]
        title_sim_idx = weight_filter(find_weight, sim_weight_array, title_sim_idx)

    get = title_array[title_sim_idx]
    tmp_origin, tmp_get = evaluate_preprocessing(origin, get)
    acc, prec = evaluate_performance(get, origin, tmp_origin, tmp_get)

    return acc, prec

def img_evaluate_model(find_title, total_df, title_array, img_index, img_vector, img_threshold = 0.1, weight_check=True):
    origin = find_answer(find_title, total_df)
    # 예측 array
    img_sim_idx, img_sim_titles, img_scores = search_range_index(find_title, title_array, img_index, img_vector, img_threshold)


    ## 2 필터링 (중량)
    if weight_check:
        find_weight = total_df[total_df['title'] == find_title].weight.iloc[0]
        if str(find_weight) == 'nan':
            weight_check = False

    if weight_check:
        sim_weight_array = total_df.weight[img_sim_idx]
        img_sim_idx = weight_filter(find_weight, sim_weight_array, img_sim_idx)


    get = title_array[img_sim_idx]

    tmp_origin, tmp_get = evaluate_preprocessing(origin, get)
    acc, prec = evaluate_performance(get, origin, tmp_origin, tmp_get)

    return acc, prec

# img -> title -> weight
def multi_filter_evaluate_model_1(find_title, total_df, title_array, img_index, img_vector, title_vector, img_thrshold=0.1, cos_threshold=0.8, weight_check=True):
    origin = find_answer(find_title, total_df)
    find_index = total_df[total_df['title'] == find_title].index[0]

    sim_index_array, sim_titles, scores = search_range_index(find_title, title_array, img_index,\
                                                             img_vector, img_thrshold)
    ## 2차 필터링용
    find_vector = title_vector[find_index]
    sim_title_vector = title_vector[sim_index_array]

    similarity_array = cal_cosine_similarity(find_vector, sim_title_vector)
    sim_title_idx = np.where(similarity_array > cos_threshold)[0]
    ## 2차 필터링용

    ## 3차 필터링 (중량)
    if weight_check:
        find_weight = total_df[total_df['title'] == find_title].weight.iloc[0]
        if str(find_weight) == 'nan':
            weight_check = False
    ## 3차 필터링 (중량)

    if len(sim_title_idx) == 0:
        if weight_check:
            sim_weight_array = total_df.weight[sim_index_array]
            sim_index_array = weight_filter(find_weight, sim_weight_array, sim_index_array)

        get = title_array[sim_index_array]
        tmp_origin, tmp_get = evaluate_preprocessing(origin, get)
        acc, prec = evaluate_performance(get, origin, tmp_origin, tmp_get)
        return acc, prec

    find_sim_title_idx = sim_index_array[sim_title_idx]

    if weight_check:
        sim_weight_array = total_df.weight[find_sim_title_idx]
        find_sim_title_idx = weight_filter(find_weight, sim_weight_array, find_sim_title_idx)

    get = title_array[find_sim_title_idx]

    tmp_origin, tmp_get = evaluate_preprocessing(origin, get)
    acc, prec = evaluate_performance(get, origin, tmp_origin, tmp_get)

    return acc, prec

# title -> image -> weight
def multi_filter_evaluate_model_2(find_title, total_df, title_array, title_index, img_vector, title_vector, title_threshold=0.1, cos_threshold=0.8, weight_check=True):
    origin = find_answer(find_title, total_df)
    find_index = total_df[total_df['title'] == find_title].index[0]

    sim_index_array, sim_titles, scores = search_range_index(find_title, title_array, title_index,\
                                                             title_vector, title_threshold)
    ## 2차 필터링용
    find_vector = img_vector[find_index]
    sim_img_vector = img_vector[sim_index_array]

    similarity_array = cal_cosine_similarity(find_vector, sim_img_vector)
    sim_img_idx = np.where(similarity_array > cos_threshold)[0]
    ## 2차 필터링용

    ## 3차 필터링 (중량)
    if weight_check:
        find_weight = total_df[total_df['title'] == find_title].weight.iloc[0]
        if str(find_weight) == 'nan':
            weight_check = False
    ## 3차 필터링 (중량)

    if len(sim_img_idx) == 0:
        if weight_check:
            sim_weight_array = total_df.weight[sim_index_array]
            sim_index_array = weight_filter(find_weight,sim_weight_array,sim_index_array)

        get = title_array[sim_index_array]
        tmp_origin, tmp_get = evaluate_preprocessing(origin, get)
        acc, prec = evaluate_performance(get, origin, tmp_origin, tmp_get)
        return acc, prec

    find_sim_title_idx = sim_index_array[sim_img_idx]


    if weight_check:
        sim_weight_array = total_df.weight[find_sim_title_idx]
        find_sim_title_idx = weight_filter(find_weight,sim_weight_array,find_sim_title_idx)

    get = title_array[find_sim_title_idx]


    tmp_origin, tmp_get = evaluate_preprocessing(origin, get)
    acc, prec = evaluate_performance(get, origin, tmp_origin, tmp_get)

    return acc, prec


# evaluate_model('푸드) 컨츄리타임 레몬향(레몬에이드) 585g', total_df, title_array, cleaned_title_array, index1, minmax_word_df1, 0.2)

##################################################################################################################################################################################################################################################
############################################################### TFIDF top_n models ###############################################################################################################################################################
##################################################################################################################################################################################################################################################

def search_tfidf_title(query_word_title,topn_title,total_df):

    reversed_query_word_title = " ".join(query_word_title.split(' ')[::-1])

    find_idx = np.where((topn_title == query_word_title) | (topn_title == reversed_query_word_title))[0]
    get = total_df.iloc[find_idx]['title'].values
    return get

def evaluate_tfidf_model(query_title,topn_title,total_df):
    find_df = total_df[total_df['title'] == query_title]
    groupNo, groupCnt = find_df['groupNo'].iloc[0], find_df['groupCnt'].iloc[0]
    query_word_title = find_df['top_2_title'].iloc[0]

    # 정답 제목
    origin = total_df[total_df['groupNo'] == groupNo].title.values

    # 예측 array
    get = search_tfidf_title(query_word_title,topn_title,total_df)

    tmp_origin, tmp_get = evaluate_preprocessing(origin, get)
    acc, prec = evaluate_performance(get, origin, tmp_origin, tmp_get)

    return acc, prec

# evaluate_tfidf_model('이엔 향신료조제품 시치미 시찌미 240g 1EA',topn_title,total_df)
