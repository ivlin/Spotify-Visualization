#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


x = 20 * np.random.rand(50)
y = 30 * np.random.rand(50)

centers = [np.random.rand(10) for i in range(5)]
cluster_std = np.random.rand(5)
X, y = make_blobs(n_samples=100, cluster_std=cluster_std, centers=centers, n_features=10, random_state=0)

x = [el[0] for el in X]
y = [el[1] for el in X]
dat = np.array(list(zip(x,y)))


# In[3]:


numerical_categories = []
dat = []

with open("code/hw1/public/spotify.csv") as csvfile:
    reader = csv.reader(csvfile)
    cols = next(reader)
    print([(ind, colname) for ind, colname in enumerate(cols)])
    first_line = next(reader)
    with open("code/hw2/spotify_numeric.csv", "w") as output:
        for ind, col in enumerate(first_line):
            try:
                _ = float(col)
                numerical_categories.append(ind)
            except:
                pass
        
        output.write(",".join([cols[ind] for ind in numerical_categories]) + "\n")
        output.write(",".join([first_line[ind] for ind in numerical_categories]) + "\n")
        dat.append([float(first_line[ind]) for ind in numerical_categories])
        for ind, line in enumerate(reader):
            output.write(",".join([line[ind] for ind in numerical_categories]) + "\n")
            dat.append([float(line[ind]) for ind in numerical_categories])

            
numerical_categories_names = [cols[ind] for ind in numerical_categories]
dat = np.array(dat)
# dat = dat / dat.max(axis=0)


# In[4]:


import math
dat = []

with open("code/hw2/spotify_numeric.csv") as csvfile:
    reader = csv.reader(csvfile)
    cols = next(reader)
    for ind, line in enumerate(reader):
        dat.append([float(val) for val in line])

min_db = min([d[4] for d in dat])
max_db = max([d[4] for d in dat])
    

dat = np.array(dat)
dat_unscaled = np.copy(dat)

# feature scaling

for d in dat:
    d[4] = math.exp(d[4])
    d[4] = d[4] - min_db
dat = np.array(dat)
    
# object normalization

dat = dat / dat.max(axis=0)
row_sums = dat.sum(axis=1)
dat = dat / row_sums[:, np.newaxis]

# K Means

CLUSTER_MIN = 2
CLUSTER_MAX = 10
label_by_cluster = {}
for cluster_count in range(CLUSTER_MIN, CLUSTER_MAX):
    clusters = KMeans(n_clusters=cluster_count, random_state=0).fit(dat)
    label_by_cluster[cluster_count] = clusters.labels_

with open("code/hw2/spotify_numeric.csv", "w") as csvfile:
    csvfile.write(",".join(cols))
    csvfile.write("," + ",".join([str(i) for i in range(CLUSTER_MIN, CLUSTER_MAX)]))
    csvfile.write("\n")
    for ind, row in enumerate(dat_unscaled):
        csvfile.write(",".join([str(el) for el in row]))
        csvfile.write("," + ",".join([str(label_by_cluster[i][ind]) for i in range(CLUSTER_MIN, CLUSTER_MAX)]))
        csvfile.write("\n")
    
with open("code/hw2/spotify_numeric_norm.csv", "w") as csvfile:
    csvfile.write(",".join(cols))
    csvfile.write("," + ",".join([str(i) for i in range(CLUSTER_MIN, CLUSTER_MAX)]))
    csvfile.write("\n")
    for ind, row in enumerate(dat):
        csvfile.write(",".join([str(el) for el in row]))
        csvfile.write("," + ",".join([str(label_by_cluster[i][ind]) for i in range(CLUSTER_MIN, CLUSTER_MAX)]))
        csvfile.write("\n")


# In[5]:


coloring = ["red", "green", "blue", "purple", "black", "pink", "orange", "teal"]
    
for dims in range(1,len(dat[0])+1):
    pca = PCA(n_components=dims, svd_solver='full')
    transformed_data = pca.fit(dat)
    
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
plt.plot(list(range(1,len(dat[0])+1)), pca.explained_variance_)

def sum_sq_loading(components, n_components, dimension, n_dimensions):
    dim_ident = np.zeros(n_dimensions)
    dim_ident[dimension] = 1
    ssl = sum(np.dot(components[:n_components], dim_ident) ** 2)
    return ssl
    
with open("code/hw2/pca_reduction.csv", "w") as f:
    f.write("Dimension,Explained variance,Explained variance ratio,")
    f.write(",".join(["{}".format(col) for col in cols]))
    f.write("\n")
    for ind, var in enumerate(pca.explained_variance_):
        f.write("{},{},{}".format(ind + 1, var, pca.explained_variance_ratio_[ind]))
        for col_ind, col in enumerate(cols):
            dim_id = np.zeros(len(cols))
            dim_id[col_ind] = 1
            f.write(",{}".format(sum_sq_loading(pca.components_, ind + 1, col_ind, len(cols))))
        f.write("\n")


# In[6]:


x = np.dot(dat, pca.components_[0])
y = np.dot(dat, pca.components_[1])
plt.scatter(x, y)


# In[7]:


# def eval_err(points, label, center, distance_fn):
#     return sum([distance_fn(point, center[label[i]]) for i, point in enumerate(points)])

# err_points = []

# for cluster_count in range(2,9):
#     clusters = KMeans(n_clusters=cluster_count, random_state=0).fit(dat)
#     err = eval_err(dat, clusters.labels_, clusters.cluster_centers_, lambda x, y: np.linalg.norm(x-y))
#     err_points.append((cluster_count, err))
    
# plt.plot([p[0] for p in err_points],
#         [p[1] for p in err_points])
# plt.show()

# coloring = ["red", "green", "blue", "purple", "black", "pink", "orange", "teal"]
# plt.scatter(x, y, c=[coloring[i] for i in clusters.labels_])
# plt.show()


# In[8]:


# Part 3 and 4 - Use unscaled data
dat = dat_unscaled


# In[9]:


from sklearn.manifold import MDS

embedding = MDS(n_components=2, random_state=0)
dat_mds = embedding.fit_transform(dat)


# In[10]:


from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

ds_eu = euclidean_distances(dat)
embedding = MDS(n_components=2, random_state=0, dissimilarity='precomputed')

print(ds_eu.shape)
dat_mds_l2 = embedding.fit_transform(ds_eu)

# with open("code/hw2/mds_euclidean.csv", "w") as f:
#     f.write("x,y")
#     f.write("," + ",".join([str(i) for i in range(CLUSTER_MIN, CLUSTER_MAX)]))
#     f.write("\n")
#     for ind, row in enumerate(dat_mds):
#         f.write(",".join([str(val) for val in row]))
#         f.write("," + ",".join([str(label_by_cluster[i][ind]) for i in range(CLUSTER_MIN, CLUSTER_MAX)]))
#         f.write("\n")


# In[11]:


ds_man = manhattan_distances(dat)
embedding = MDS(n_components=2, random_state=0, dissimilarity='precomputed')

dat_mds_l1 = embedding.fit_transform(ds_man)

dat_mds = np.concatenate([dat_mds_l2, dat_mds_l1], axis=1)

with open("code/hw2/mds.csv", "w") as f:
    f.write("x_l2,y_l2,x_l1,y_l1")
    f.write("," + ",".join([str(i) for i in range(CLUSTER_MIN, CLUSTER_MAX)]))
    f.write("\n")
    for ind, row in enumerate(dat_mds):
        f.write(",".join([str(val) for val in row]))
        f.write("," + ",".join([str(label_by_cluster[i][ind]) for i in range(CLUSTER_MIN, CLUSTER_MAX)]))
        f.write("\n")


# In[12]:


from scipy.stats import pearsonr

dat = []

with open("code/hw2/spotify_numeric.csv") as csvfile:
    reader = csv.reader(csvfile)
    cols = next(reader)
    for ind, line in enumerate(reader):
        # ignore the cluster coloring columns
        dat.append([float(val) for col, val in enumerate(line) if not cols[col].isnumeric()]) 
dat = np.array(dat)

attr_matrix = [[] for el in dat[0]]

for ind, _ in enumerate(dat[0]):
    for _, row in enumerate(dat):
        attr_matrix[ind].append(row[ind])
# print(len(attr_matrix))

corr_matrix = [[0 for j in attr_matrix] for i in attr_matrix]

for i in range(len(attr_matrix)):
    for j in range(len(attr_matrix)):
        corr_matrix[i][j] = corr_matrix[j][i] = abs(pearsonr(attr_matrix[i], attr_matrix[j])[0])

corr_matrix = np.array(corr_matrix)

embedding = MDS(n_components=2, random_state=0, dissimilarity='precomputed')
dat_mds = embedding.fit_transform(corr_matrix)

with open("code/hw2/correlation_mds.csv", "w") as f:
    f.write("labels,x,y\n")
    for ind, row in enumerate(corr_matrix):
        f.write(",".join([cols[ind]] + [str(val) for val in row]))
        f.write("\n")


# In[ ]:




