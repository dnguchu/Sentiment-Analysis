# analysis_helpers.py
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def mean_tfidf_difference(X, y):
    pos_mask = (y==1)
    neg_mask = (y==0)
    mean_pos = np.asarray(X[pos_mask,:].mean(axis=0)).ravel()
    mean_neg = np.asarray(X[neg_mask,:].mean(axis=0)).ravel()
    return mean_pos - mean_neg

def word_cosine_matrix_from_X(X_train, feature_names, words):
    # returns DataFrame of cosine similarity for words present in feature_names
    word2idx = {w:i for i,w in enumerate(feature_names)}
    cols = [word2idx[w] for w in words if w in word2idx]
    present_words = [w for w in words if w in word2idx]
    if len(cols)==0:
        raise ValueError("No words matched feature names")
    W = X_train[:, cols].toarray().T
    cos = cosine_similarity(W)
    return pd.DataFrame(cos, index=present_words, columns=present_words)
