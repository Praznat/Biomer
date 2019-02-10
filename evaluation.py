import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def nonoverlap(xy, labels, j=5):
    knn = NearestNeighbors(n_neighbors=j, algorithm='ball_tree').fit(xy)
    distances, indices = knn.kneighbors(xy)
    li = labels[indices]
    my_labels = li[:,0][:, np.newaxis]
    neighbor_labels = li[:,1:]
    frac_equal = np.equal(my_labels, neighbor_labels).mean(axis=1)
    return np.mean(frac_equal)
def calc_accuracy(df):
    v = df.iloc[:,1:]
    p = v.std()
    return p
def calc_entropy(p):
    entropy = 0
    if p > 0:
        entropy += p * np.log(p)
    if p < 1:
        entropy += (1 - p) * np.log(1 - p)
    return entropy
def calc_info_entropy(df):
    v = df.iloc[:,1:]
    p = v.sum() / v.count()
    return p.apply(calc_entropy)
def entropy(z, labels):
    cz = pd.concat([pd.DataFrame({"c":labels}), pd.DataFrame(z)], axis=1)
    entropy = cz.groupby("c").apply(calc_info_entropy).as_matrix().sum()
    accuracy = cz.groupby("c").apply(calc_accuracy).as_matrix().mean()
    return entropy, accuracy


