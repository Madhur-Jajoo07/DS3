import pandas as pd
import numpy as np
import math 
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection as sk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from tabulate import tabulate

#defining a function to normalise data
def normalise(y):
    x = list(y)
    m = min(x)
    n = max(x)
    for i in range(len(x)):
        x[i] = (x[i] - m)/(n-m)
    return(x)


# reading the csv
csv = pd.read_csv("SteelPLateFaults-2class.csv", delimiter=",")

#getting list of all attribute names
att = list(csv.columns)

#dividing data on basis of class
by_class = csv.groupby('Class')
data0 = by_class.get_group(0)
data1 = by_class.get_group(1)


#splitting data using train_test_split
data0_train , data0_test , data0_lable_train  , data0_lable_test = sk.train_test_split(data0 , data0['Class'] , test_size=0.3 , random_state=42)
data1_train , data1_test , data1_lable_train  , data1_lable_test = sk.train_test_split(data1 , data1['Class'] , test_size=0.3 , random_state=42)

#converting to data frames
data0_test = pd.DataFrame(data0_test)
data0_train= pd.DataFrame(data0_train)
data1_test = pd.DataFrame(data1_test)
data1_train= pd.DataFrame(data1_train)

#merging the data frames using concat function of pandas
train_frames = [data0_train,data1_train]
test_frames = [data0_test,data1_test]
train_label_frame = [data0_lable_train,data1_lable_train]
test_label_frame = [data0_lable_test,data1_lable_test]

data_train = pd.concat(train_frames)
data_test = pd.concat(test_frames)
data_labels_train = pd.concat(train_label_frame)
data_labels_test = pd.concat(test_label_frame)

# saving the data in a new csv for future reference 
data_train.to_csv('SteelPlateFaults-train.csv',index=False)
data_test.to_csv('SteelPlateFaults-test.csv',index=False)

#reading the data for question no.1
print("**********question 1**********")
data_train = pd.read_csv('SteelPlateFaults-train.csv')
data_test = pd.read_csv('SteelPlateFaults-test.csv')
scaler = StandardScaler()
scaler.fit_transform(data_test)
scaler.fit_transform(data_train)

#using the KNeighborsClassifier function from sklearn for classifer 
for k in [1,3,5]:
    
    classifer = KNeighborsClassifier(n_neighbors=k)
    classifer.fit(data_train.values,data_labels_train)
    prediction = classifer.predict(data_test.values)
    matrix = confusion_matrix(data_labels_test,prediction)
    accuracy = accuracy_score(data_labels_test,prediction)
    print("****confusion matrix of non normalised data for k=",k,"\n",matrix)
    print("percentage accuracy score for non normalised data for k =",k,"is",round(accuracy*100,3))
 

#question 2
print("**********question 2**********")
# reading csv files
data_train_to_normalise = pd.read_csv('SteelPlateFaults-train.csv')
data_test_to_normalise = pd.read_csv('SteelPlateFaults-test.csv')

data_normalised_train = pd.DataFrame()
# normalising the data using the function defines above
for i in range(len(att)-1):    
    data_normalised_train [att[i]] =normalise(data_train_to_normalise[att[i]])

# saving the normalised data to csv file 
data_normalised_train.to_csv('SteelPlateFaults-train-Normalised.csv', index=False)

#reading the normalised data
data_normalised_train = pd.read_csv('SteelPlateFaults-train-Normalised.csv')

#normalising the test data
data_normalised_test = pd.DataFrame()
for i in range (len(att)-1) :
    m = min(data_train[att[i]])
    n = max(data_train[att[i]])
    l = []
    for j in range (len(data_test_to_normalise[att[i]])): 
        l.append((data_test_to_normalise[att[i]][j] - m) / (n-m))
    data_normalised_test [att[i]] = l

#saving the data to a csv
data_normalised_test.to_csv('SteelPlateFaults-test-Normalised.csv', index=False)
#reading the csv
data_normalised_test = pd.read_csv('SteelPlateFaults-test-Normalised.csv')
#reading the class from the non normalised data so as to use in confusion matrix
a = data_test_to_normalise['Class']
b = data_train_to_normalise['Class']

scaler.fit_transform(data_normalised_test)
scaler.fit_transform(data_normalised_train)
#using the KNeighborsClassifier function from sklearn for classifer 
for k in [1,3,5]:
    
    classifer = KNeighborsClassifier(n_neighbors=k)
    classifer.fit(data_normalised_train.values, b.values)
    #classifer.fit(data_normalised_train,data_normalised_train['Class'])
    prediction = classifer.predict(data_normalised_test.values)
    matrix = confusion_matrix(a.values,prediction)
    accuracy = accuracy_score(a.values,prediction)
    print("****confusion matrix of normalised data for k=",k,"\n",matrix)
    print("percentage accuracy score of normalised data for k=",k,"is",round(accuracy*100,3))


#q3
print("**********question 3**********")
# reading the non normalised splitted data from the csv saved above
data_train = pd.read_csv('SteelPlateFaults-train.csv')
data_test = pd.read_csv('SteelPlateFaults-test.csv')

#dropping the columns because of which likelihood may be equal to 0 and may give singularity matrix error
data_test.drop(columns=['X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'], inplace=True)
data_train.drop(columns=['X_Minimum','Y_Minimum','TypeOfSteel_A300','TypeOfSteel_A400'], inplace=True)
# getting the list of columns
att = list(data_train.columns)
#dividing data on basis of class
by_class = data_train.groupby('Class')
data_train_0 = by_class.get_group(0)
data_train_1 = by_class.get_group(1)
data_train_1 = data_train_1.drop('Class',axis=1)
data_train_0 = data_train_0.drop('Class',axis=1)
test_class = data_test['Class']
data_test = data_test.drop('Class',axis = 1)

# getting the mean vectors
mean_class0 = []
mean_class1 = []
for i in range (23):
    mean0 = data_train_0[att[i]].mean()
    mean_class0.append(mean0)
    mean1 = data_train_1[att[i]].mean()
    mean_class1.append(mean1)


#getting the covarience matrix
cov_class0 = np.cov(data_train_0.values.T)
cov_class1 = np.cov(data_train_1.values.T)

# defining the function to calculate likelihood
def likelihood(x, mean, cov):
    power = -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), (x - mean))
    a =  (np.exp(power)) / ((2 * np.pi)**12.5 * (abs(np.linalg.det(cov)))**0.5)
    return(a)
#calculating prior
prior_class0 = len(data_train_0) / (len(data_train))
prior_class1 = len(data_train_1) / (len(data_train))

#calculating the bayes prediction
prediction =[]
for i in data_test.itertuples(index= False):
    y_0 = (likelihood(np.array(i), mean_class0,cov_class0 )*prior_class0 ) 
    y_1 = (likelihood(np.array(i), mean_class1,cov_class1 )*prior_class1 ) 
    if (y_0 > y_1):
        prediction.append(0)
    if (y_1 > y_0):
        prediction.append(1)
#calculating the confusion matrix and accuracy 
matrix = confusion_matrix(test_class , prediction)
print("confusion matrix of bayes classifer is \n ",matrix)
accuracy = (matrix[0][0] + matrix[1][1]) / (matrix[0][0] + matrix[1][1] + matrix[0][1] + matrix[1][0])
print("percentage accuracy of bayes classifer is\n" , round(accuracy*100 , 3))


# getting the covarience matrix to a excel file so as to put their screen shots in the report
cov_class0 = pd.DataFrame(cov_class0)
round(cov_class0,3).to_excel('covarience matrix of class 0.xlsx')
cov_class1= pd.DataFrame(cov_class1)
round(cov_class1,3).to_excel('covarience matrix of class 1.xlsx')

#tabulate
d = [ ["KNN", 89.614],
     ["KNN-normalized", 97.329 ],
     ["Bayes Classifer", 94.362]]

print(tabulate(d, headers=["ML Algorithm","Accuracy in %"]))

