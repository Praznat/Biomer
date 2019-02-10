import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
import pystan
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.neighbors import NearestNeighbors, kneighbors_graph, radius_neighbors_graph
import evaluation

def gca(x, y, Z, n_biomes, xymag=1, zmag=1, constrained=True, plot=False, **kwargs):
    start = time.time()
    results = {"n_biomes":n_biomes}
    z = zmag * Z
    x = -x / (2 * np.std(x))
    y = -y / (2 * np.std(y))
    
    if not constrained:
        xyz = np.transpose(np.vstack((xymag * x, xymag * y, np.transpose(z))))
        kmeans = KMeans(n_clusters=n_biomes, random_state=0).fit(xyz)
        labels = kmeans.labels_
    
    if constrained:
        xy = np.transpose(np.vstack((x, y)))
        xyz = np.transpose(np.vstack((xymag * x, xymag * y, np.transpose(z))))
        n_neighbors=3
        knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(xy)
        distances, indices = knn.kneighbors(xy)
        knn_graph = kneighbors_graph(xy, n_neighbors, include_self=False)
        linkage = "ward"
        aggclust = AgglomerativeClustering(linkage=linkage, connectivity=knn_graph, n_clusters=n_biomes)
        aggclust.fit(xyz)
        labels = aggclust.labels_

    if kwargs.get("voronoi"):
        ldf = pd.DataFrame({"c":labels, "x":x, "y":y})
        cc = ldf.groupby("c").mean().as_matrix()
        vor = Voronoi(cc)
        fig = voronoi_plot_2d(vor, show_points=True, show_vertices=False)
    
    colors = None
    if kwargs.get("show_z") and (ztype == "kills" or ztype == "levels"):
        if kwargs.get("agg"):
            kdf = pd.DataFrame({"c":labels, "z":z})
            killfreq = kdf.groupby("c").mean()["z"]
            kflo, kfhi = (killfreq.min(), killfreq.max())
            killfreqZ = (killfreq - kflo) / (kfhi - kflo)
            pkf = killfreqZ.loc[labels]
            C = np.vstack((pkf.values, np.zeros_like(pkf.values), np.zeros_like(pkf.values)))
            C = np.transpose(C)
        else:
            C = np.vstack(((z - np.min(z))/(np.max(z) - np.min(z)), np.zeros_like(z), np.zeros_like(z)))
            C = np.transpose(C)
    else:
        colors = cm.rainbow(np.linspace(0, 1, np.max(labels) + 1))
        C = colors[labels]
        
    end = time.time()
    results["time"] = end - start
    results["nonoverlap"] = evaluation.nonoverlap(np.transpose(np.vstack((x, y))), labels, j=5)
    
    ent, prob = evaluation.entropy(Z, labels)
    results["entropy"] = ent
    results["highprob"] = prob
    
    if plot:
        plt.scatter(x, y, s=1, color=C, alpha=0.5)
        if colors is not None:
            handles = []
            for c in range(n_biomes):
                handles.append(mpatches.Patch(color=colors[c], label=groups[c]))
            if kwargs.get("legend"):
                plt.legend(handles=handles, loc='best', prop={'size': 6})
        plt.xlim(x.min(), x.max())
        plt.ylim(y.min(), y.max())
        plt.show()
    return results

def gaa(x, y, Z, n_regions, n_biomes, method="kmeans", plot=False, **kwargs):
    results = {"n_biomes":n_biomes}
    start = time.time()
    x = -x / (2 * np.std(x))
    y = -y / (2 * np.std(y))
    xy = np.transpose(np.vstack((x, y)))
    
    rn_graph = radius_neighbors_graph(xy, 2.0 / n_regions, mode='connectivity', include_self=True)
    zm = []
    for i in range(len(x)):
        neighbors = rn_graph.getrow(i).nonzero()[1]
        zm.append(Z[neighbors].mean(axis=0))
    zpc = np.array(zm)


    if method == "kmeans":
        kmeans = KMeans(n_clusters=n_biomes, random_state=0).fit(zpc)
        labels = kmeans.labels_
    elif method == "agglomerative":
        linkage = "ward"
        aggclust = AgglomerativeClustering(linkage=linkage, n_clusters=n_biomes)
        aggclust.fit(zpc)
        labels = aggclust.labels_
    else:
        raise ValueError("nope")
    
    groups = np.unique(labels)
    
    end = time.time()
    results["time"] = end - start
    results["nonoverlap"] = evaluation.nonoverlap(np.transpose(np.vstack((x, y))), labels, j=5)
    
    ent, prob = entropy(Z, labels)
    results["entropy"] = ent
    results["highprob"] = prob
    
    if plot:
        colors = cm.rainbow(np.linspace(0, 1, np.max(labels) + 1))
        C = colors[labels]
        plt.scatter(x, y, s=1, color=C, alpha=0.5)
        if colors is not None:
            handles = []
            for c in range(len(np.unique(labels))):
                handles.append(mpatches.Patch(color=colors[c], label=groups[c]))
            if kwargs.get("legend"):
                plt.legend(handles=handles, loc='best', prop={'size': 6})
        plt.xlim(x.min(), x.max())
        plt.ylim(y.min(), y.max())
        plt.show()
    return results