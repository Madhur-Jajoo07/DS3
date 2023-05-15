# Madhur Jajoo
# B20211
# 7597389137


# lab5 part:- B

from numpy.core.fromnumeric import mean
import pandas as pd
import numpy as np
import math 
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.preprocessing import PolynomialFeatures


# reading csv
data = pd.read_csv("abalone.csv")

#splitting data
data_train, data_test = train_test_split(data, test_size=0.3, random_state=42,shuffle=True)

# making csv of the train and the test data
data_train = pd.DataFrame(data_train)
data_test = pd.DataFrame(data_test)
data_train.to_csv('abalone-train.csv')
data_test.to_csv('abalone-test.csv')

# getting the column names 
att = data.columns

# question 1
print("***** question 1 *****")

# finding the pearson correlation between rings and all other attributes and the attribute with whome the rings have the maximum correlation
corr = {}
max = -1
for i in range (len(att)):
    if(att[i] != "Rings"):
        a , _ = pearsonr(data[att[i]],data["Rings"])
        corr[i] = a
    if(a>max):
        max= a 
        attribute_index=i
# attribute is the attribute with which rings have maximum correlation
attribute=att[attribute_index]
print("The Highest value of Pearsons correlation coefficient of rings is with", attribute)

#modeling linerar regression

X = data_train[attribute].values.reshape(-1,1)
Y = data_train["Rings"].values.reshape(-1,1)
reg = LinearRegression().fit(X,Y)
Y_prediction = reg.predict(X)
#A
print("***** question_1 sub part A *****")

#plotting the scatter plot of the training data and best fit line
plt.scatter(X,Y,alpha=0.75, marker= '+',label='Training Data')    
plt.plot(X,Y_prediction,color='black',label='Best Fit Line')
plt.grid()
plt.ylabel("Rings")
plt.xlabel(attribute)
plt.legend()
plt.show()
#B
print("***** question_1 sub part B *****")
# RMSE accuracy of training data 
b = ((mean((Y-Y_prediction)**2))**0.5)
print("prediction accuracy using RMSE: ",b)
#C
print("***** question_1 sub part C *****")
# RMSE accuracy of test data
x = data_test[attribute].values.reshape(-1,1)
y = data_test["Rings"].values.reshape(-1,1)
reg1 = LinearRegression().fit(x,y)
y_prediction = reg1.predict(x)
print("RMSE accuraacy on: ",(mean((y-y_prediction)**2))**0.5)                          
#D
print("***** question_1 sub part D *****")

#plotting the scater plot of no. of rings
plt.scatter(y,y_prediction, label = 'On Test data')
plt.grid()
plt.xlabel("Actual Rings")
plt.ylabel("Predicted Rings")
plt.legend()
plt.show()

# QUESTION - 2
print("***** question_2 *****")
#A
print("***** question_2 sub part A *****")

# getting the data 
Y1 = data_train["Rings"]
X1 = data_train.drop(["Rings"],axis = 1)
# modeling the linear regression
reg2 = LinearRegression().fit(X1,Y1)
y_pred = reg2.predict(X1)
print("RMSE on train data",(mean((Y1-y_pred)**2))**0.5)

#B
print("***** question_2 sub part B *****")
# getting data
Y1 = data_test["Rings"]
X1 = data_test.drop(["Rings"],axis = 1)
# modeling the linear regression
reg2 = LinearRegression().fit(X1,Y1)
y_pred = reg2.predict(X1)
print("RMSE on test data",mean(((Y1-y_pred)**2))**0.5)
#C
print("***** question_2 sub part C *****")

#plotting the scatter plot of number of rings
plt.scatter(Y1,y_pred, label = 'On Test data')
plt.grid()
plt.xlabel("Actual Rings")
plt.ylabel("Predicted Rings")
plt.legend()
plt.show()

# q3
print("***** question_3 *****")
x_train=data_train.iloc[:,:-1].values
y_train=data_train.iloc[:,7].values
x_test=data_test.iloc[:,:-1].values
y_test=data_test.iloc[:,7].values

x=list(data_train[attribute])


# defining a function for simple non linear regression
def Simple_NLR(p,x_given,y_given):
    z=[]
    for i in range(len(x)):
        l=[]
        for j in range(p+1):
            l.append(math.pow(x[i],j))
        z.append(l)
    a=np.dot(np.transpose(z),z)
    b=np.linalg.inv(a)
    c=np.dot(b,np.transpose(z))
    w=np.dot(c,y_train)
    y_pred=[]
    for i in x_given:
        yt=0
        for j in range(len(w)):
            yt=yt+w[j]*math.pow(i,j)
        y_pred.append(yt)
    s=0
    for i in range(len(y_given)):
        s=s + (y_given[i]-y_pred[i])*(y_given[i]-y_pred[i])
    E_rmse=math.sqrt(s/len(y_given))
    return E_rmse,w


print("***** question_3 sub part A *****")

print("Prediction accuracies for Training data")

# calculating simple non linear regression for different polynomial degrees for train data
p2,w2=Simple_NLR(2,list(data_train[attribute]),y_train)
p3,w3=Simple_NLR(3,list(data_train[attribute]),y_train)
p4,w4=Simple_NLR(4,list(data_train[attribute]),y_train)
p5,w5=Simple_NLR(5,list(data_train[attribute]),y_train)
for i in range(4):
    print("RMSE for p = ",i+2," : ",Simple_NLR(i+2,list(data_train[attribute]),y_train)[0])

# plotting the bar graphs
plt.bar([2,3,4,5],[p2,p3,p4,p5])
plt.yticks([x/100 for x in range(239,252)])
plt.ylim([2.4,2.51])
plt.xlabel("p")
plt.ylabel("Prediction Accuracy")
plt.show()

print("***** question_3 sub part B *****")

print("Prediction accuracy for Test data")
#print(len(list(data_test[attribute])))
x=list(data_train[attribute])

# calculating simple non linear regression for different polynomial degrees for test data
p2,w2=Simple_NLR(2,list(data_test[attribute]),y_test)
p3,w3=Simple_NLR(3,list(data_test[attribute]),y_test)
p4,w4=Simple_NLR(4,list(data_test[attribute]),y_test)
p5,w5=Simple_NLR(5,list(data_test[attribute]),y_test)
for i in range(4):
    print("RMSE for p = ",i+2," : ",Simple_NLR(i+2,list(data_test[attribute]),y_test)[0])
# plotting the bar graphs
plt.bar([2,3,4,5],[p2,p3,p4,p5])
plt.yticks([x/100 for x in range(239,252)])
plt.ylim([2.38,2.45])
plt.xlabel("p")
plt.ylabel("Prediction Accuracy")
plt.show()

print("***** question_3 sub part C *****")


x=list(data_train[attribute])
p5,w5=Simple_NLR(5,list(data_train[attribute]),y_train)
xt= np.linspace(0, 1, 1000)

# getting the curve
yt=w5[0] + w5[1]*(xt) + w5[2]*(xt**2) + w5[3]*(xt**3) + w5[4]*(xt**4) + w5[5]*(xt**5)
#plotting the scatter plot on which the fit curve will be plotted
plt.scatter(x,y_train)
#plotting the curve on the above scatter plot
plt.plot(xt,yt,'r')
plt.xlabel(attribute)
plt.ylabel("Rings")
plt.show()

print("***** question_3 sub part D *****")

# getting the no. of rings
y_pred_test=[]
p2,w2=Simple_NLR(2,list(data_test[attribute]),y_test)
for i in list(data_test[attribute]):
    yt=0
    for j in range(2):
        yt=yt+w2[j]*math.pow(i,j)
    y_pred_test.append(yt)
# plotting the no. of rings
plt.scatter(y_test,y_pred_test)
plt.xlabel("Actual Rings")
plt.ylabel("Predicted Rings")
plt.show()


# q4

print("***** question_4 *****")

# defining a function for the multiple non linear regression
def Multiple_NLR(p,x_given,y_given):
    poly_features = PolynomialFeatures(p) #p is the degree
    x_poly = poly_features.fit_transform(x_train)
    regressor=LinearRegression()
    regressor.fit(x_poly,y_train)
    x_poly_given=poly_features.fit_transform(x_given)
    y_pred = regressor.predict(x_poly_given)
    s=0
    for i in range(len(y_given)):
        s=s + (y_given[i]-y_pred[i])*(y_given[i]-y_pred[i])
    E_rmse=math.sqrt(s/len(y_given))
    return E_rmse,y_pred

print("***** question_4 sub part A *****")

print("Prediction accuracies of Train data")
# calculating multiple non linear regression for different polynomial degrees for train data
p2,y2=Multiple_NLR(2,x_train,y_train)
p3,y3=Multiple_NLR(3,x_train,y_train)
p4,y4=Multiple_NLR(4,x_train,y_train)
p5,y5=Multiple_NLR(5,x_train,y_train)
for i in range(4):
    print("Prediction Accuracy for p = ",i+2," : ",Multiple_NLR(i+2,x_train,y_train)[0])
# plotting the bar graphs
plt.bar([2,3,4,5],[p2,p3,p4,p5])
plt.yticks([x/100 for x in range(160,210,5)])
plt.ylim([1.55,2.15])
plt.xlabel("p")
plt.ylabel("Prediction Accuracy")
plt.show()

print("***** question_4 sub part B *****")

print("Prediction accuracies of Test data")
# calculating multiple non linear regression for different polynomial degrees for test data
p2,y2=Multiple_NLR(2,x_test,y_test)
p3,y3=Multiple_NLR(3,x_test,y_test)
p4,y4=Multiple_NLR(4,x_test,y_test)
p5,y5=Multiple_NLR(5,x_test,y_test)
for i in range(4):
    print("Prediction Accuracy for p = ",i+2," : ",Multiple_NLR(i+2,x_test,y_test)[0])
# plotting the bar graphs
plt.bar([2,3,4,5],[p2,p3,p4,p5])
plt.xlabel("p")
plt.ylabel("Prediction Accuracy")
plt.show()

print("***** question_4 sub part C *****")

p2,y_pred=Multiple_NLR(2,x_test,y_test)
plt.scatter(y_test,y_pred)
plt.xlabel("Actual Rings")
plt.ylabel("Predicted Rings")
plt.show()

