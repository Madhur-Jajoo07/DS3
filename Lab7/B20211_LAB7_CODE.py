#%%
#importing modules
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN


#reading the data
data = pd.read_csv('Iris.csv' , delimiter=',')

# making a list for the species column in numeric form
species = []
for i in range (len(data['Species'])):
    if (data['Species'][i] == 'Iris-setosa'):
        species.append(0)
    if (data['Species'][i] == 'Iris-versicolor'):
        species.append(1) 
    if (data['Species'][i] == 'Iris-virginica'):
        species.append(2)
# dropping the species column
data_to_pca = data.drop('Species' , axis=1)
att = data_to_pca.columns
# performing the PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_to_pca)

print('question 1')
# calculating the eigen values and eigen vectors
x =data_to_pca
eigenval,eigenvec= np.linalg.eig(np.cov(data_to_pca.T))
#plotting 
c = np.linspace(1,4,4)
plt.bar(c,[round(i,3) for i in eigenval])
plt.xticks(np.arange(min(c), max(c)+1, 1.0))
plt.xlabel('eigen values')
plt.ylabel('no. of components')
plt.title('Eigenvalue vs. components')
plt.show()

print()
print('question2')
# performing the Kmeans clusturing on the reduced data
K = 3
kmeans = KMeans(n_clusters=K)
b = kmeans.fit(data_pca)
label = kmeans.fit_predict(data_pca)
centers = kmeans.cluster_centers_
#plotting the cluster
label_names =['Iris-setosa','Iris-versicolor','Iris-virginica','centres']
for i in range(3):
    colour=['red', 'blue','pink']
    filtered_label0 = data_pca[ label == i]
    plt.scatter(filtered_label0[:,0] , filtered_label0[:,1],  color = colour[i] )
plt.scatter([centers[i][0] for i in range (3)] , [centers[i][1] for i in range (3)] , color = 'black')
plt.legend(label_names)
plt.title(' K-means (K=3) clustering')
plt.show()

print()
print('question 2 part b')
# distortion measure
print('distortion measure kmeans for k=3', round(b.inertia_ , 3))

print()
print('question 2 part c')
#defining a function to calculate purity score 
def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)
    #print(contingency_matrix)
    # Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    # Return cluster accuracy
    return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)

print("the purity score is:", round(purity_score(species,label),3))

print()
print('question 3')
# performing the Kmeans clustering fro different values of k
l=[]
p_score =[]
K = [2,3,4,5,6,7]
for i in K:
    kmeans = KMeans(n_clusters=i)
    b = kmeans.fit(data_pca)
    label = kmeans.fit_predict(data_pca)
    l.append(b.inertia_)
    p_score.append(round(purity_score(species , label),3))
# plotting the graph of distortion measure wrt values of k 
plt.plot(l,K)
plt.ylabel('no. of clusters')
plt.xlabel('distortion measure')
plt.title('Number of clusters(K) vs. distortion measure')
plt.show()
print('the purity scores for values of k are : ' ,p_score)

print()
print('question 4')
# GMM clustering 
k = 3
gmm = GaussianMixture(n_components = k)
gmm.fit(data_pca)
label = gmm.predict(data_pca)
#plotting the cluster
centers = gmm.means_
for i in range(3):
    colour=['red', 'blue','pink']
    filtered_label0 = data_pca[ label == i]
    plt.scatter(filtered_label0[:,0] , filtered_label0[:,1],  color = colour[i])
plt.scatter([centers[i][0] for i in range (3)] , [centers[i][1] for i in range (3)] , color = 'black')
plt.legend(label_names)
plt.title('gmm clustering for k = 3')
plt.show()
# calculating the liklihood
print()
print('question 4 part b')
x = gmm.lower_bound_
print('distortion measure for gmm clustering ' , x*(len(data_to_pca[att[1]])))

print('question 4 part c')
print('the purity score for gmm clustering',round(purity_score(species,label),3))

# performing the gmm for different no. of clusters
print('question 5')

liklihood = []
p_score_gmm =[]
for i in K:
    gmm = GaussianMixture(n_components = i , random_state=5)
    gmm.fit(data_pca)
    label = gmm.predict(data_pca)
    liklihood.append(gmm.lower_bound_ * len(data_to_pca[att[1]]))
    p_score_gmm.append(round(purity_score(species , label),3))
plt.plot(liklihood , K)
plt.ylabel('no. of clusters')
plt.xlabel('distortion measure')
plt.title('Number of clusters(K) vs. distortion measure')
plt.show()
print(p_score_gmm)

#getting the species in the data frame for dbscan
def DBSCAN_(ep , samples):
    dbscan_model = DBSCAN(eps=ep, min_samples=samples).fit(data_pca)
    return dbscan_model.labels_

eps = [1, 1, 5, 5]
min_samples = [4, 10, 4, 10]
for i in range(4):
    dbscan_model = DBSCAN(eps=eps[i], min_samples=min_samples[i]).fit(data_pca)
    DBSCAN_predictions = dbscan_model.labels_
    print(f'Purity score for eps={eps[i]} and min_samples={min_samples[i]} is',round(purity_score(species, DBSCAN_predictions), 3))
    plt.scatter(data_pca[:,0], data_pca[:,1], c=DBSCAN_predictions, cmap = 'rainbow', s=15)
    plt.title(f'DBSCAN for eps={eps[i]} and min_samples={min_samples[i]}')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
