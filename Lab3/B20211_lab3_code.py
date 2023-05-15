#Madhur Jajoo
#B20211
#mobile no. 7597389137

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics as st
from sklearn.decomposition import PCA
#defining function for mean

def mea(a):
    b = st.mean(a)
    return(b)
# defining function for standard deviation
def std(a):
    b = st.stdev(a)
    return(b)
#defining a function for median
def media(a):
    x=st.median(a)
    return x
#defining a function to detect outliers
def outliers(x):
    
    x.sort(reverse= False)
    if len(x) %2 == 0 :
        q1= x[round(len(x) *0.25)]
        q3 = x[round(len(x)*0.75)]
    else:
        q1= x[round((len(x) +1) *0.25)]
        q3 = x[round((len(x) +1)*0.75)]
    iqr = q3 - q1 
    outlierLst = [] 
    for i in range(len(x)):
        if (x[i] < q1 - 1.5*iqr or x[i] > q3 + 1.5*iqr):
            outlierLst.append(i)
        else:
            continue
    return outlierLst
#cdefining a function for the calculation of root mean square error
def rmse(x,y):  
    return (sum(((x-y)**2).sum(axis=1))/len(x))**0.5
#reading csv file
data = pd.read_csv("pima-indians-diabetes.csv",delimiter=',')
# getting a list containing name of attributes
att = list(data.columns)


print("**********q1_a**************")
#declaring lists for min and max to be able to print in a specific format
maxlst =[]
minlst =[]
maxlstn =[]
minlstn =[]

# running a loop for detecting and replacing outliers and after thet normalising
for i in range (8):
    x = list(data[att[i]])
    med = media(x)
    a = outliers(x)
    #print("before removing outliers",len(outliers(x)))
    #replacing outliers with median
    for j in (a):
        z = x[j]
        data[att[i]] = data[att[i]].replace({z: med})
        x[j] = med
    #gettinh values for min and max
    m = max(x)
    n = min(x)
    #print("after removing outliers",len(outliers(x)))
    #appending values in lists to print them in a proper manner
    maxlst.append(m)
    minlst.append(n)
    normalized =[]
    #normalising the data
    for k in range (len(x)):
        normalized.append((((x[k] - n)/(m - n))*7 ) +5)
    # appending new min and new max in lists
    maxlstn.append(max(normalized))
    minlstn.append(min(normalized))
#copying data for question no.3
data1 = data.copy()
data2 = data.copy()





# q1_b
print("**********q1_b**************")
#declaring lists for mean and std_dev to be able to print in a specific format
meanlst =[]
stdlst =[]
meanlstn =[]
stdlstn =[]
for i in range(8):
    #appending values of mean and standard deviation in lists
    meanlst.append(mea(data[att[i]]))
    stdlst.append(std(data[att[i]]))
    m = mea(data[att[i]])
    s = std(data[att[i]])
    standarized = []
    x = list(data[att[i]])
    #standarizeng the data 
    for j in range (len(x)):
        standarized.append((x[j] - m)/s)
        x[j] = ((x[j] - m)/s)
    #appending new values of mean and standard deviation in lists
    meanlstn.append(mea(standarized))
    stdlstn.append(std(standarized))
# printing the data obtained above
print()
for i in range (8):
    print("maximum before normalization of ", att[i],"is",maxlst[i] )
    print("maximum after normalization of ", att[i],"is",maxlstn[i] )
    print()
for i in range (8):
    print("minimum before normalization of ", att[i],"is",minlst[i] )
    print("minimum after normalization of ", att[i],"is",minlstn[i] )
    print()
for i in range (8):
    print("mean before standrization of ", att[i],"is",round(meanlst[i] ,3) )
    print("mean after standrization of ", att[i],"is",round(meanlstn[i],3) )
    print()
for i in range (8):
    print("standard deviation before standrization of ", att[i],"is",round(stdlst[i],3) )
    print("standard seviation after standrization of ", att[i],"is",round(stdlstn[i],3) )
    print()

#q2
print("**********q2**************")
print("**********q2_a**************")
mean =[0,0]
covar =[[13,-3],[-3, 5]]
# getting 2-D synthetic data 
d =np.random.multivariate_normal(mean,covar,1000,'ignore').T
print("distribution: ",d)
#plotting a scatter plot of data samples with titles and named axes
plt.title("Scatter plot of 2D Synthetic data(Q2_a)")   
plt.xlabel("x1")
plt.ylabel("x2")
plt.scatter(d[0], d[1], marker= '+')
plt.show()

#getting eigenvalues and eigenvectors
print("**********q2_b**************")
eigenvalues,eigenvectors = np.linalg.eig(np.cov(d))
print("eigenvalues are : " , eigenvalues)
print()
print("eigenvectors are: ", eigenvectors)
eig_vec1 =eigenvectors[:,1]
eig_vec2 =eigenvectors[:,0]
o = [0,0]
# plotting scater plot of eigen directions and then plotting the eigen vectors with title and named axes
plt.title("Scatter plot for 2D Synthetic data and eigen directions(Q2_b)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.scatter(d[0], d[1], marker='+')
plt.quiver(o,o, eig_vec1, eig_vec2, scale =10)
plt.show()

#projectng data on first eigenvector with title and named axes
a = np.dot(d.T , eig_vec1)
plt.scatter(d[0], d[1], marker='+')
plt.title("projected value on 1st eigen directions(Q2_c_i")   
plt.xlabel("x1")
plt.ylabel("x2")
plt.quiver(o,o, eig_vec1, eig_vec2, scale =10)
plt.scatter(a*eigenvectors[0][1],a*eigenvectors[1][1] , color = 'red',marker='.')
plt.show()
print("**********q2_c**************")
#projecting data on second eigenvector with title named axes
a = np.dot(d.T , eig_vec2)
plt.title("projected value on 2nd eigen directions(Q2_c_ii)")   
plt.xlabel("x1")
plt.ylabel("x2")
plt.scatter(d[0], d[1],marker='+')
plt.quiver(o,o, eig_vec1, eig_vec2, scale =10)
plt.scatter(a*eigenvectors[0][0],a*eigenvectors[1][0] , color = 'red', marker='.')
plt.show()

#getting reconstruction error using mean square error
print("**********q2_d**************")
s1 =np.dot(d.T,eigenvectors)
s2 =np.dot(s1,eigenvectors.T)
error =np.square(np.subtract(d.T,s2)).mean()   
print("the error is",error)

print("**********q3**************")
#standarizing the data copied above
data1  =  (data1 - data1.mean())/data1.std()
#dropping the class attribute from data
data1 = data1.drop(['class'], axis=1)

print("**********q3_a**************")
#principle component analysis
pca =PCA(n_components=2)
pca.fit(data1)
data_trfd =pca.transform(data1)
#plotting scatter plot of 2-D transformed data withe title and names axes
plt.xlabel("1st principle comp.")
plt.ylabel("2nd principle comp.")
plt.title("Reduced dimesional data")
plt.scatter(data_trfd[:,0],data_trfd[:,1], marker='+', color = 'purple') 
plt.show()

#finding eigen values and vectors using function of numpy
x =data1
eigenval_x,eigenvec_x= np.linalg.eig(np.cov(x.T))   
print(eigenval_x)
print(eigenvec_x)
u =pca.components_
v =pca.explained_variance_
#printing eigenvalues and variance
print("Eigen value of PC 1:",eigenval_x[0])
print("Vairance of PC 1:",v[0])
print("Eigen value of PC 2:",eigenval_x[1])
print("Variance of PC 2:",v[1])

print("**********q3_b**************")
#declaring a list for eigen values
eigenlst =[]
for i in eigenval_x:
    eigenlst.append(i)
#rearranging the list containing eigen values in descending order
eigenlst.sort(reverse=True) 
#plotting all the eigen values   
plt.bar(range(1,9),eigenlst)   
plt.xlabel("Number of eigen Value")
plt.ylabel("Eigen Value")
plt.show()

print("**********q3_c**************")
# calculation of root mean square error of the transformed data
x2 =pca.inverse_transform(data_trfd)
print(rmse(x,x))
rmselst=[]

for i in range(1,8):
    pca =PCA(n_components=i)
    pca.fit(x)
    xp =pca.transform(x)
    xn =pca.inverse_transform(xp)
    rmselst.append(rmse(x,xn))
print(rmselst)
plt.bar(range(1,8),rmselst)
plt.plot(range(1,8),rmselst,color='green')
plt.scatter(range(1,8),rmselst)
plt.xlabel("components")
plt.ylabel("Root mean square error")
plt.show()

# getting covariance matrix for all the values of l
for i in range (1,9):
    pca = PCA(n_components=i)
    pcm = pca.fit_transform(data1)
    cov_mat = np.dot(pcm.T,pcm)
    print("covarience matrix for l =",i, cov_mat)

print("**********q3_d**************")
print("covarience matrix of original data")

print(data1.cov())


