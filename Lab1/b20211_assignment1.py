# Madhur Jajoo
#B20211
#mobile no. 7597389137

import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np

#reading csv file

data = pd.read_csv("pima-indians-diabetes.csv",delimiter=',')


#assigining the attributes their values
pregnant = data['pregs']
plasma = data['plas']
pressure = data['pres']
skin = data['skin']
test = data['test']
bmi = data ['BMI']
pedigree = data['pedi']
age = data['Age']
cls = data['class']
# declaring a list for all the attributes to use in loop
att = [pregnant,plasma,pressure,skin,test,bmi,pedigree,age,cls]
# declaring the list of atributes as string
att_str = ['pregnant','plasma','pressure','skin','test','bmi','pedigree','age']

# Question 1

#mean mode median minimum maximum standard deviation
for i in range (8):
    
    x = att[i]
#mean
    totalp =0
    list = []
    b=0
    o=0
    n=0
    for j in range (768) :
        totalp = totalp + x[j]
        b=x[j]
    # list for the next questions
        list.append(b)
    #sum for standard deviation
        o= o + x[j]*x[j]
        n= n+ x[j]
        j= j+1
    print("mean of", att_str[i],"is",round(totalp/768,3))
#median
    list.sort(reverse=True)
    c=list[383]
    d= list[384]
    print ("median of", att_str[i],"is",round((c+d)/2,3))
# mode 
    L=[]
    k= 0
    while k < len(list) :
        L.append(list.count(list[k]))
        k += 1
    d1 = dict(zip(list, L))
    d2={l for (l,v) in d1.items() if v == max(L) }
    print("Mode(s) of :" , att_str[i], "is/are  "+ str(d2))
#minimum
    print("minimum of ", att_str[i],"is",list[767])
#maximum
    print("maximum of ", att_str[i],"is",list[0])
#standard deviation
    print("standard deviation of ", att_str[i],"is",round(((o/768)-((n/768)*(n/768)))**0.5,3))
    print()
    print()
    i = i+1

#Question 2

# Scatter plots

# with age
for i in range (7):
    str= "age vs "+att_str[i]
    plt.title(str)
    plt.xlabel("age")
    plt.ylabel(att_str[i])
    plt.scatter(age,att[i])
    i+=1
    plt.show()

# with BMI
for i in [0,1,2,3,4,6,7]:
    str= "bmi vs "+att_str[i]
    plt.title(str)
    plt.xlabel("bmi")
    plt.ylabel(att_str[i])
    plt.scatter(bmi,att[i])
    i+=1
    plt.show()

#question 3

#correaltion
#for age and all attributes
for j in range (8):
    x = att[j]
    sum_x = 0
    sum_y =0
    sum_xy = 0
    sum_x2=0
    sum_y2 =0
    for i in range (768):
        sum_x = sum_x + x[i]
        sum_y = sum_y + age[i]
        sum_xy= sum_xy + (x[i])*(age[i])
        sum_x2 =  sum_x2 + (x[i])*(x[i])
        sum_y2 = sum_y2 + (age[i])*(age[i])
        i = i+1
    print("correlation of age and ",att_str[j]," is",round((768 * sum_xy - sum_x * sum_y)/(((768 * sum_x2 - sum_x * sum_x)* (768 * sum_y2 -  sum_y * sum_y))**0.5),3))
    print()
    j= j+1
    
#for bmi and all attributes
for j in range (8):
    x = att[j]
    sum_x = 0
    sum_y =0
    sum_xy = 0
    sum_x2=0
    sum_y2 =0
    for i in range (768):
        sum_x = sum_x + x[i]
        sum_y = sum_y + bmi[i]
        sum_xy= sum_xy + (x[i])*(bmi[i])
        sum_x2 =  sum_x2 + (x[i])*(x[i])
        sum_y2 = sum_y2 + (bmi[i])*(bmi[i])
        i = i+1
    print("correlation of bmi and ",att_str[j]," is",round((768 * sum_xy - sum_x * sum_y)/(((768 * sum_x2 - sum_x * sum_x)* (768 * sum_y2 -  sum_y * sum_y))**0.5),3))
    print()
    j= j+1

#Question 4

# HISTOGRAMS
df=pd.DataFrame(skin)
df.hist()
df=pd.DataFrame(pregnant)
df.hist()
plt.show()

#question 5

#Groupby
getting_class = data.groupby("class")
c1 = getting_class.get_group(1)
c0 = getting_class.get_group(0)
pregsclass1 = [(c1["pregs"])]
pregsclass2 = [(c0["pregs"])]
#plotting the histogram
plt.hist(pregsclass1,edgecolor= 'black',color= 'yellow')
plt.xlabel("Values")
plt.ylabel("Pregnants of class1")
plt.title("Histogram For Pregnants with class 1")
plt.show()
plt.hist(pregsclass2,edgecolor= 'red',color= 'black')
plt.xlabel("Values")
plt.ylabel("Pregnants of class2")
plt.title("Histogram For Pregnants with class 0")
plt.show()

#Question 6
# plotting box plots
for i in range (8):
    plt.title(att_str[i])
    plt.boxplot(att[i])
    i = i+1
    plt.show()

















