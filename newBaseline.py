"""
@version 09/16/2019
@author: Gerry Dozier
modifier: Linyuan Zhang
baseline with one of three models
"""

import Data_Utils
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import StratifiedKFold
import numpy as np
#from sklearn import preprocessing
#from sklearn.model_selection import cross_val_score

def calculate_acc(mask):
    """
    CU_X, Y = Data_Utils.Get_Casis_CUDataset()
    #CU_X = []
    temp = []
    for i in range(100):
        for j in range(95):
            CU_X[i][j] = CU_X[i][j] * mask[i][j]
    """
    #temp = np.zeros([100])
    temp = []
    for te in range(100):
       temp.append(mask)
    temp = np.array(temp)
    CU_X, Y = Data_Utils.Get_Casis_CUDataset()
    #CU_X = []
    for i in range(100):
        for j in range(95):
            CU_X[i][j] = CU_X[i][j] * temp[i][j]

    #rbfsvm = svm.SVC(gamma = 'auto')
    lsvm = svm.LinearSVC()
    #mlp = MLPClassifier(max_iter=2000)

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=0)
    fold_accuracy = []

    scaler = StandardScaler()
    tfidf = TfidfTransformer(norm=None)
    dense = Data_Utils.DenseTransformer()

    for train, test in skf.split(CU_X, Y):
        #train split
        CU_train_data = CU_X[train]
        train_labels = Y[train]

        #test split
        CU_eval_data = CU_X[test]
        eval_labels = Y[test]

        # tf-idf
        tfidf.fit(CU_train_data)
        CU_train_data = dense.transform(tfidf.transform(CU_train_data))
        CU_eval_data = dense.transform(tfidf.transform(CU_eval_data))

        # standardization
        scaler.fit(CU_train_data)
        CU_train_data = scaler.transform(CU_train_data)
        CU_eval_data = scaler.transform(CU_eval_data)

        # normalization
        CU_train_data = normalize(CU_train_data)
        CU_eval_data = normalize(CU_eval_data)

        train_data =  CU_train_data
        eval_data = CU_eval_data

        # evaluation
        #rbfsvm.fit(train_data, train_labels)
        lsvm.fit(train_data, train_labels)
        #mlp.fit(train_data, train_labels)

        #rbfsvm_acc = rbfsvm.score(eval_data, eval_labels)
        lsvm_acc = lsvm.score(eval_data, eval_labels)
        #mlp_acc = mlp.score(eval_data, eval_labels)

        #fold_accuracy.append((lsvm_acc, rbfsvm_acc, mlp_acc))

        # rbfsvm to be chosen as the model
        return lsvm_acc
