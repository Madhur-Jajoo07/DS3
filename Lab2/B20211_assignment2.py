import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.core.dtypes.missing import notnull
import statistics as st
from array import *

#Reading the csv files using python
data_miss = pd.read_csv("landslide_data3_miss.csv", delimiter=",")
data_org = pd.read_csv("landslide_data3_original.csv", delimiter=",")

att = list(data_miss.columns)
# defining a function to find how many missing values are available 
def missing(df):
    sum = df.isnull().sum()
    return sum
#defining a function for mean
def mea(a):
    x=sum(a)/len(a)
    return x
#defining a function for median
def media(a):
    x=st.median(a)
    return x
#defining a function for mode
def mod(a):
    x=st.mode(a)
    return x
#defining a function for standard deviation
def stdev(a):
    x=st.stdev(a)
    return x
# defining a function to find where are n=missing values available 
def pos_mis(a):
    x=np.where(pd.isnull(a))
    return x
#defining a function to calculate rmse
def rmse(a,b):
    sum = 0
    for i in range(len(a)):
        c=a[i]
        d=b[i]
        e=(c-d)**2
        sum = sum + e
    x = (sum/len(a))**0.5
    return x



#question 1 
print()
print("************ q1 ************")
print()
for j in range (len(att)):
    print("no. of missing values in ", att[j]," is ", missing(data_miss[att[j]]) )

#bar graph
fig = plt.figure(figsize=(15,10))
plt.xlabel("attributes")
plt.ylabel("No. of missing values")
for i in range (len(att)):
    plt.bar(att[i],missing(data_miss[att[i]]))
plt.show()

#q2_a
print()
print("************ q2_A ************")
print()
print("lab viva")
print(missing(data_miss['temperature']))
print(missing(data_miss['pressure']))
print()
print()
mis_stat = pos_mis(data_miss[att[1]])   # getting the columns having no station id
#removing the rows with no station id
for i in mis_stat :
    data_new =  data_miss.drop(i)
print("Q2_A no. of tuples deleted is  ",len(data_miss[att[1]]) - len(data_new[att[1]]))

#q2_b
print()
print("************ q2_B ************")
print()

#deleting the rows with 3 or more values unavailable
data_dropped3 = data_new.dropna(axis = 0 , thresh = 7 )


print("Q2_B no. of tuples deleted is :", len(data_new[att[2]])-len(data_dropped3[att[2]]))

#question 3
print()
print("************ q3 ************")
print()

tot = 0
for j in range (len(att)):
    print("no. of missing values in ", att[j]," is ", missing(data_dropped3[att[j]]) )
    tot += missing(data_dropped3[att[j]])
print("no. of total missing values is",tot)


#q4
print()
print("************ q4_a_i ************")
print()
y=[]
for i in range (len(att)):
    x= pos_mis(data_dropped3[att[i]])
    a= list(x)
    y.append(a)
# duplicationg csv for q5
data_dropped3.to_csv("data_drop3.csv",index=False, sep=',')
data_drop3 = pd.read_csv("data_drop3.csv",delimiter=',')

# filling mean in the attributes 
for i in range (len(att)-2):
    data_dropped3[att[i+2]].fillna(data_dropped3[att[i+2]].mean() , inplace = True)
    print("no. of missing values in ", att[i]," is ", missing(data_dropped3[att[i]]) )

print()
print("lab viva")
print("new",missing(data_dropped3['temperature']))
print("new",missing(data_dropped3['pressure']))
print()
print()
#printing mean mode median 
for i in range (len(att)-2):
    print()
    print("mean of ",att[i+2],"in processed file is",mea(data_dropped3[att[i+2]]), "and in original file it is ",mea(data_org[att[i+2]]))
    print("median of ",att[i+2],"in processed file is",media(data_dropped3[att[i+2]]), "and in original file it is ",media(data_org[att[i+2]]))
    print("mode of ",att[i+2],"in processed file is",mod(data_dropped3[att[i+2]]), "and in original file it is ",mod(data_org[att[i+2]]))
    print("standard deviation of ",att[i+2],"in processed file is",stdev(data_dropped3[att[i+2]]), "and in original file it is ",stdev(data_org[att[i+2]]))
    print()
print()
print("************ q4_a_ii ************")
print()
#calculating rmse
for i in range (len(att)-2):
    l=0
    k=0
    m=np.array(data_org[att[i+2]])
    n=np.array(data_dropped3[att[i+2]])
    for j in range (len(y[i+2])):
        g=y[i+2]
        l=list(n[g[j]])
        k=list(m[g[j]])
    print("the root mean square error (RMSE) between the original and replaced values for",att[i+2],"is", rmse(l,k))
    plt.bar(att[i+2],rmse(l,k))
plt.xlabel("attributes")
plt.ylabel("rmse values")
plt.show()

print()
print("************ q4_b_i ************")
print()

data_drop3 = data_drop3.interpolate()
#printinh mean mode median
for i in range (len(att)-2):
    print()
    print("mean of ",att[i+2],"in processed file is",mea(data_drop3[att[i+2]]), "and in original file it is ",mea(data_org[att[i+2]]))
    print("mode of ",att[i+2],"in processed file is",mod(data_drop3[att[i+2]]), "and in original file it is ",mod(data_org[att[i+2]]))
    print("median of ",att[i+2],"in processed file is",media(data_drop3[att[i+2]]), "and in original file it is ",media(data_org[att[i+2]]))
    print("standard deviation of ",att[i+2],"in processed file is",stdev(data_drop3[att[i+2]]), "and in original file it is ",stdev(data_org[att[i+2]]))
    print()
print()
print("************ q4_b_ii ************")
print()
# claculating rmse
for i in range (len(att)-2):
    l=0
    k=0
    m=np.array(data_org[att[i+2]])
    n=np.array(data_drop3[att[i+2]])
    for j in range (len(y[i+2])):
        g=y[i+2]
        l=list(n[g[j]])
        k=list(m[g[j]])
    print("the root mean square error (RMSE) between the original and replaced values for",att[i+2],"is", rmse(l,k))
    plt.bar(att[i+2],rmse(l,k))
plt.xlabel("attributes")
plt.ylabel("rmse values")
plt.show()

#question 5
print()
print("************ Q5 ************")
print()
"""for i in range (len(att)-2):
    plt.title(att[i+2])
    plt.boxplot(data_drop3[att[i+2]])
    plt.show()"""
def outliers(y):
    x = list(y)
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
# outliers of rain and temperature
out_temp = list(outliers(data_drop3['temperature']))
out_rain = list(outliers(data_drop3['rain']))
print(out_rain)
print(out_temp)
plt.boxplot(out_temp)
plt.show()
plt.boxplot(out_rain)
plt.show()



# replacing with median
med_temp = media(data_drop3['temperature'])
med_rain = media(data_drop3['rain'])
y = data_drop3['temperature']
z= data_drop3['rain']
for i in out_temp :
    y[i] = med_temp
for i in out_rain:
    z[i] = med_rain


out_temp = list(outliers(data_drop3['temperature']))
out_rain = list(outliers(data_drop3['rain']))
plt.boxplot(out_temp)
plt.show()
plt.boxplot(out_rain)
plt.show()

