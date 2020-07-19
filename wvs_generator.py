import multiprocessing
from collections import Counter

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import dok_matrix

import pickle
import numpy as np

from tqdm import tqdm

class WVSGenerator():
    
    def __init__(self, w2v_model=None, tfidf_model=None, vector_size=100, avg=True, norm=True):
        
        self.scaler = MinMaxScaler()
        
        self.vector_size = vector_size
        self.avg = avg
        self.norm = norm
        
        self.tfidf_feature_names = None
        self.total_wvs = None
        
        self.w2v_model = w2v_model
        self.tfidf_model = tfidf_model
        
    def get_total_wvs(self):
        return self.total_wvs
        
    def save_total_wvs(self, save_path):
        print('=> save total_wvs to {0}'.format(save_path))
        np.savez(save_path, self.total_wvs)

    def get_w2v_model(self):
        return self.w2v_model
    
    def save_w2v_model(self, save_path):
        print('=> save w2v_model to {0}'.format(save_path))
        self.w2v_model.save(save_path)
        
    def get_tfidf_model(self):
        return self.tfidf_model
    
    def save_tfidf_model(self, save_path):
        print('=> save tfidf_model to {0}'.format(save_path))
        pickle.dump(self.tfidf_model, open(save_path, "wb"))
        
    def _normalize(self, vector):

        vector = vector.reshape(-1, 1)
        normalized_vector = self.scaler.fit_transform(vector)
        normalized_vector = normalized_vector.reshape(1, -1)
        
        return normalized_vector
        
    def _make_w2v_model(self, arr_title, n_worker, window_size=5, min_count=1):
        print('=> start training w2v model....')
        splited_token_list = [str_token.split(' ') for str_token in arr_title]
        w2v_model = Word2Vec(splited_token_list, size=100, window=window_size, min_count=min_count, workers=n_worker)
        print('=> end training w2v model....')
        self.w2v_model = w2v_model
        
    def _make_tfidf_model(self, arr_title):
        print('=> start training tfidf model....')
        vectorizer = TfidfVectorizer(encoding=u'utf-8', token_pattern='[가-힣a-zA-Z0-9]+', lowercase=False, min_df=1)
        vectorizer.fit_transform(arr_title)
        print('=> end training tfidf model....')
        self.tfidf_model = vectorizer
    
    def _get_tfidf_feature_names(self):
        return np.array(self.tfidf_model.get_feature_names())
    
    def _find_word_weight(self, query_title):
        query_sparse_matrix = self._get_query_sparse_matrix(query_title)
        sparse_dict = self._make_sparse_dict(query_sparse_matrix)

        if len(sparse_dict) == 0:
            words, weights = np.array([]), np.array([])
            return words, weights

        # key & value
        keys = np.array([i[1] for i in sparse_dict.keys()])
        words = self.tfidf_feature_names[keys]
        weights = np.array(list(sparse_dict.values()))

        return words, weights

    def _get_word_vector(self, word):
        
        try:
            word_vector = self.w2v_model.wv[word]
            
        except KeyError:
            word_vector = np.zeros(self.vector_size)
            
        return word_vector
    
    def _get_query_sparse_matrix(self, query_title):
        return self.tfidf_model.transform(np.array([query_title]))
    
    def _make_sparse_dict(self, sparse_matrix):
        return dok_matrix(sparse_matrix)
    
    def _wvs_process(self, arr_title, process_num=0):
        
        num = 0
        total_count = len(arr_title)
        
        arr_wvs = np.zeros(shape=(total_count, self.vector_size))

        for query_title in tqdm(arr_title):
            wvs = self.make_wvs_vector(query_title)
            arr_wvs[num] = wvs
            num += 1

        return arr_wvs
    
    # make feature vector
    def make_wvs_vector(self, query_title):
                
        word_array, weight_array = self._find_word_weight(query_title)
        
        weighted_word_dict = dict(zip(word_array, weight_array))
        weighted_word_dict = sorted(weighted_word_dict.items())
        
        wvs_vector = np.zeros(shape=self.vector_size)
        
        for word, weight in weighted_word_dict:
            word_vector = self._get_word_vector(word)
            weighted_vector = weight * word_vector
            wvs_vector += weighted_vector
        
        if len(weighted_word_dict):
            denominator = len(weighted_word_dict)
        else:
            denominator = 1e-13
            
        if self.avg:
            wvs_vector = wvs_vector / denominator
        
        if self.norm:
            wvs_vector = self._normalize(wvs_vector)
        
        return wvs_vector
    
    def do(self, arr_title, n_worker):

        # make w2v model
        if self.w2v_model is None:
            self._make_w2v_model(arr_title, n_worker=n_worker)
        
        # make tfidf model
        if self.tfidf_model is None:
            self._make_tfidf_model(arr_title)
        
        # get tfidf feature names()
        self.tfidf_feature_names = self._get_tfidf_feature_names()
        
        if n_worker > 1:
            
            # split arr with n chunks
            arr_process_num = [i for i in range(0, n_worker)]
            arr_chunks = np.array_split(arr_title, n_worker)
            
            # multi-processing
            print('{0} process running...'.format(n_worker))
            pool = multiprocessing.Pool(processes = n_worker)
            results = pool.starmap(self._wvs_process, zip(arr_chunks, arr_process_num))
            pool.close()
            pool.join()

            total_wvs = np.vstack(results)
        
        else:
            total_wvs = self._wvs_process(arr_title)
            
        self.total_wvs = total_wvs
        
        print('=> make_wvs process is done....')
        
        return total_wvs
    
if __name__ == '__main__':
    
    # test training wvs generator
    str_token_list = ['아버지 가방에 들어가신다', '이 문장은 예시 입니다.']
    
    wvs_generator = WVSGenerator()
    total_wvs = wvs_generator.do(str_token_list, n_worker=3)
    
    # make title vector
    test_title = '아버지 가방에 들어가신다'
    title_vector = wvs_generator.make_wvs_vector(test_title)

    print(title_vector)
    
# output (vector_size):
# [[0.61880634 0.27024904 0.44064757 0.83468468 0.65102677 0.35300063
#   0.37036863 0.54567909 0.57597407 0.36810817 1.         0.86062455
#   0.67168928 0.42570229 0.5331886  0.72507176 0.30209711 0.71503442
#   0.60365092 0.35338949 0.31571771 0.49931365 0.20970246 0.28141216
#   0.6110637  0.68429382 0.25311739 0.62917164 0.29522942 0.23894072
#   0.50450487 0.41283829 0.28094866 0.59555713 0.95990337 0.46211074
#   0.68893765 0.71982497 0.35049874 0.64356724 0.56462712 0.17448475
#   0.63351981 0.20482745 0.21019455 0.47121575 0.99900673 0.38729769
#   0.63209597 0.52264756 0.84319659 0.41244818 0.8066209  0.13496225
#   0.61388895 0.43092339 0.22324945 0.52286587 0.43005578 0.78749165
#   0.53311584 0.25554156 0.13834877 0.7977924  0.69759237 0.59366645
#   0.37241615 0.3126012  0.99450398 0.15049812 0.51263346 0.41689056
#   0.82039361 0.88730487 0.52170603 0.26739176 0.62206429 0.75206031
#   0.52906226 0.32147562 0.61910118 0.60525264 0.06223111 0.707124
#   0.83785665 0.67957613 0.17868843 0.34992439 0.75228178 0.64574129
#   0.         0.33884197 0.4195472  0.26002865 0.42838927 0.53568893
#   0.65819303 0.50582355 0.37990875 0.62522273]]