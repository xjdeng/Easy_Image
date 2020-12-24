from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from path import Path as path

cols = [str(i) for i in range(128)]

def cluster(df, n):
    k = KMeans(n_clusters=n)
    return k.fit_predict(df[cols])

def smart_cluster_old(df, min_n, max_n):
    best_n, best_s = (1, -2)
    g = df[cols]
    for i in range(min_n, max_n + 1):
        model = KMeans(n_clusters = i)
        test = model.fit_predict(g)
        score = silhouette_score(g, test)
        if score > best_s:
            best_n, best_s = (i, score)
    model = KMeans(n_clusters = best_n)
    return model.fit_predict(g)

def run_old(df, dest, min_n = 2, max_n = 10, prefix = ""):
    path(dest).mkdir_p()
    if min_n == max_n:
        clusters = cluster(df, min_n)
    else:
        clusters = smart_cluster_old(df, min_n, max_n)
    for i in range(max(clusters) + 1):
        path("{}/{}".format(dest, i)).mkdir_p()
    for c,f in zip(clusters, df['file']):
        myfile = path(prefix + f)
        myfile.copy("{}/{}".format(dest, c))
        
def run(df, dest, eps = 0.6, prefix = ""):
    path(dest).mkdir_p()
    d = DBSCAN(metric="euclidean", eps=eps)
    clusters = d.fit_predict(df[cols]) + 1
    for i in range(max(clusters) + 1):
        path("{}/{}".format(dest, i)).mkdir_p()
    for c,f in zip(clusters, df['file']):
        myfile = path(prefix + f)
        myfile.copy("{}/{}".format(dest, c))
            