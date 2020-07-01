import pandas as pd
import numpy as np
import pickle
import operator
import multiprocessing
from tqdm import tqdm

from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import dok_matrix



class WVSGenerator():
    
    def __init__(self, arr_title, cleaned_arr_title, w2v_model, tfidf_model):
        
        self.arr_title = arr_title
        self.cleaned_arr_title = cleaned_arr_title
        self.scaler = MinMaxScaler()
        
        self.w2v_model = w2v_model
        self.tfidf_model = tfidf_model
        self.tfidf_feature_names = np.array(self.tfidf_model.get_feature_names())
        
        self.vector_size = self.w2v_model.vector_size
        
        self.total_wvs = None

    def get_total_wvs(self):
        return self.total_wvs
        
    def save_total_wvs(self, save_path):
        np.savez(save_path, self.total_wvs)
        
    def do(self, n_worker):
        
        if n_worker > 1:
            
            # split arr with n chunks
            arr_process_num = [i for i in range(0, n_worker)]
            arr_chunks = np.array_split(self.arr_title, n_worker)
            arr_cleaned_chunks = np.array_split(self.cleaned_arr_title, n_worker)
            

            # multi-processing
            print('{0} process running...'.format(n_worker))
            pool = multiprocessing.Pool(processes = n_worker)
            results = pool.starmap(self._wvs_process, zip(arr_chunks, arr_cleaned_chunks, arr_process_num))
            pool.close()
            pool.join()

            total_arr_wvs = np.vstack(results)
        
        else:
            total_arr_wvs = self._wvs_process(self.arr_title, self.cleaned_arr_title)
            
        self.total_wvs = np.ascontiguousarray(total_arr_wvs).astype('float32')
        
        return total_arr_wvs

    # make feature vector
    def make_wvs_vector(self, query_title, arr_title=[], cleaned_arr_title=[]):
        
        if arr_title == []:
            arr_title = self.arr_title
            cleaned_arr_title = self.cleaned_arr_title
        
        word_array, value_array = self._find_word_weight(query_title, arr_title, cleaned_arr_title)
        
        weighted_word_dict = dict(zip(word_array, value_array))
        weighted_word_dict = sorted(weighted_word_dict.items())
        
        wvs_vector = np.zeros(shape=self.vector_size)
        
        for word, value in weighted_word_dict:
            w2v_value = self.w2v_model.wv[word]
            weighted_value = w2v_value * value
            wvs_vector += weighted_value
        
        denominator = len(weighted_word_dict)
        if not len(weighted_word_dict):
            denominator = 1e-13

        wvs_vector = wvs_vector / denominator
        normalized_wvs_vector = self._normalize(wvs_vector)
        
        return normalized_wvs_vector
    
    def _wvs_process(self, arr_title, cleaned_arr_title, process_num=0):
        
        num = 0
        total_count = len(arr_title)
        
        arr_wvs = np.zeros(shape=(total_count, self.vector_size))

        for query_title in tqdm(arr_title):
            wvs = self.make_wvs_vector(query_title, arr_title, cleaned_arr_title)
            arr_wvs[num] = wvs
            num += 1

        return arr_wvs

    def _find_word_weight(self, query_title, arr_title, cleaned_arr_title):

        # find title inx & cleaned title
        query_idx = np.where(arr_title == query_title)[0].item(0)
        cleaened_query_title = cleaned_arr_title[query_idx]
        
        query_sparse_matrix = self.tfidf_model.transform(np.array(cleaened_query_title))
        sparse_dict = dok_matrix(query_sparse_matrix)

        if len(sparse_dict) == 0:
            words, values = np.array([]), np.array([])
            return words, values

        # key & value
        keys = np.array([i[1] for i in sparse_dict.keys()])
        words = self.tfidf_feature_names[keys]
        values = np.array(list(sparse_dict.values()))

        return words, values
    
    def _normalize(self, vector):

        vector = vector.reshape(-1, 1)
        normalized_vector = self.scaler.fit_transform(vector)
        normalized_vector = normalized_vector.reshape(1, -1)
        
        return normalized_vector
    
if __name__ == '__main__':
    
    arr_title = pd.read_csv('example/new_feat_df.csv', usecols=['title']).values
    cleaned_arr_title = pd.read_csv('example/new_feat_df.csv', usecols=['cleaned_title']).fillna('').values
    
    # Word2vec & TFIDF model path
    WVS_PATH = 'Model/가공식품_word2vec.model'
    TFIDF_PATH = 'Model/가공식품_tfidf_model.pkl'
    
    w2v_model = Word2Vec.load(WVS_PATH)
    tfidf_model = pickle.load(open(TFIDF_PATH, 'rb'))
    
    vector_generator = WVSGenerator(arr_title, cleaned_arr_title, w2v_model, tfidf_model)
    
    query_title = '산고추 고추절임 업소용식자재 (500gX10개) 한푸드'
    
    query_vector = vector_generator.make_wvs_vector(query_title)
    
    print(query_vector)
    
# output (vector_size):
# [[0.7993041  0.45954755 0.43287463 0.58399381 0.62696965 0.41444568
#   0.42715278 0.21784938 0.42107257 0.39600341 0.6753512  0.43215121
#   0.28987078 0.30139453 0.77971103 0.51240434 0.34315612 0.37671802
#   0.70597699 0.51440302 0.18105755 0.47145197 0.69664769 0.53683441
#   0.55223246 0.76233321 0.79113157 0.81572251 0.35385517 0.22793472
#   0.40146304 0.83502754 0.38259046 0.65550829 0.48707275 0.58369225
#   0.8558958  0.86365299 0.         0.54324413 0.2838306  0.27718614
#   0.91887353 0.213611   0.47868564 0.2926584  0.49688589 0.62966477
#   0.32772961 0.28971896 0.45733867 0.81319629 0.35194272 0.49208292
#   0.33863438 0.5860372  0.78709779 0.32629157 0.83702437 0.43670857
#   0.51970286 0.58142041 0.7283164  0.44165328 0.69414472 0.38550232
#   0.36747124 0.62724117 0.78538031 0.62944537 0.63595775 0.53387729
#   0.51067102 0.20312996 0.3784164  0.61668457 0.57342853 0.77173496
#   0.44978332 0.75412803 0.29874427 0.62084654 0.59570668 0.72516136
#   0.38966769 0.59565013 0.54689843 0.25979465 0.69620666 0.97684997
#   0.42726998 0.78741461 0.3665031  0.1980029  0.52462379 0.34604349
#   0.56335754 0.40334404 0.74219414 0.73507773 0.55447879 0.57526841
#   0.7830574  0.73007993 0.38000325 0.64353878 0.61818209 0.73507546
#   0.55633269 0.27911975 0.75582588 0.47228369 0.57053293 0.29146557
#   0.24709219 0.55689494 0.4397716  0.31814666 0.0784147  0.54194603
#   0.59058249 0.30320304 0.55516541 0.59461697 0.50318493 0.25268175
#   0.63755497 0.51927957 0.66201459 0.74739778 0.32071416 0.84807033
#   0.10462778 0.63066664 0.27378149 0.52960732 0.63209641 0.38921925
#   0.44095743 0.46289351 0.96286715 0.61159595 0.38923528 0.7298151
#   0.81325945 0.51796488 0.59248876 0.62451137 0.6509626  0.64513702
#   0.39773623 0.71520807 0.43713605 0.3789452  0.53417452 0.50131186
#   0.41800101 0.83702052 0.4764972  0.77813945 0.54409759 0.78756508
#   0.24054322 0.75215056 0.22881333 0.46260653 0.66104794 0.35638452
#   0.6797639  0.26633944 0.48571695 0.45446518 0.34336551 0.60379169
#   0.89973002 0.52658932 0.32903472 0.36746675 0.39542926 0.66817701
#   0.50686835 0.63019941 0.49982627 0.60891389 0.58907521 0.40274162
#   0.70191907 0.66471756 0.48172845 1.         0.66104881 0.1861169
#   0.36615752 0.46711602 0.40663426 0.2299847  0.72481804 0.53904228
#   0.44282582 0.69629679]]: