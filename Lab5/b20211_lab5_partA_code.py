# Madhur Jajoo
# B20211
# 7597389137


# lab5 part:- A

import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# reading the csv files
data_test = pd.read_csv('SteelPlateFaults-test.csv')
data_train = pd.read_csv('SteelPlateFaults-train.csv')

# droppping the attributes mentioned
data_train = data_train.drop(columns=['X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'], axis=1)
data_test = data_test.drop(columns=['X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'], axis=1)

# getting class in other variables and then dropping it from the data sets
by_class = data_train.groupby('Class')
data_train_0 = by_class.get_group(0)
data_train_1 = by_class.get_group(1)
data_train_c0 = data_train_0['Class']
data_train_0 = data_train_0.drop(['Class'], axis=1)
data_train_c1 = data_train_1['Class']
data_train_1 = data_train_1.drop(['Class'], axis=1)
data_test_c = data_test['Class']
data_test = data_test.drop(['Class'], axis= 1)


for q in [2,4,8,16]:

    # modeling the GMM
    gmm_c0 = GaussianMixture(n_components=q , covariance_type='full', reg_covar= 1e-4)
    gmm_c0 .fit(data_train_0.values)
    gmm_c1 = GaussianMixture(n_components=q , covariance_type='full', reg_covar= 1e-4)
    gmm_c1.fit(data_train_1.values)
    prediction = []

    a = gmm_c0.score_samples(data_test.values)
    b = gmm_c1.score_samples(data_test.values)

    for i in range (len(a)):
        if a[i] > b[i] :
            prediction.append(0)
        if a[i] < b[i] :
            prediction.append(1)
    # getting the confusion matrix
    matrix = confusion_matrix (data_test_c.values, prediction)
    # getting the accuracy score
    accuracy = accuracy_score(data_test_c.values, prediction)
    print("confusion matrix for q = " , q, "is\n" , matrix)
    print("accuracy score for q = ",q, "is :" , accuracy)




