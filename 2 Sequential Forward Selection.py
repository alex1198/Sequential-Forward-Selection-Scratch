#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# In[2]:


df = pd.read_csv("absent.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.dtypes


# In[5]:


df.isnull()


# In[6]:


df.isnull().sum()


# In[7]:


dataset = df.values # convert into 2D array
print(dataset)


# In[8]:


m = dataset[:,:-1] 
n = dataset[:,-1] # target column
num_of_columns = m.shape[1] # number of columns
print(num_of_columns)
num_of_cols_seven_five = int((75*num_of_columns)/100)# take only 75% columns from total columns
print(num_of_cols_seven_five)


# In[9]:


numberFold = 5 
labels = np.unique(n) # unique labels in target column
numOfLabels = len(labels) # number of unique labels 
print(numOfLabels)


# In[10]:


def kfold(dataset):
    classify = list()
    rowsPerFold = list()
    for unik in labels: # iterate all unique labels
        rows = dataset[np.where(n==unik)] # take "unik" label data 
        np.random.shuffle(rows) # shuffle rows
        classify.append(rows) 
        l = len(rows)
        rowsPerFold.append(int(l/numberFold)) # calculating the percent of labels in every fold
    return classify,rowsPerFold


# In[11]:


def stratified(dataset):
    classify, rowsPerFold = kfold(dataset)
    datasetPartition = list()
    a = 0
    # maintain the percent of labels in every fold
    while(a<numberFold): # iterate on number of fold
        fold = list()
        b = 0
        while(b<numOfLabels): # iterate to fold on percent of labels
            if(a==numberFold-1):
                x = classify[b][p:]
                fold.extend(x) 
            else:
                p = rowsPerFold[b] * a
                q = rowsPerFold[b] * (a+1)                
                x = classify[b][p:q]
                fold.extend(x)
            b = b + 1
        np.random.shuffle(fold)
        datasetPartition.append(fold)
        a = a + 1
    return datasetPartition


# In[12]:


validateData = stratified(dataset) # data with stratification
print(validateData)


# In[13]:


def firstFeatureSelect(x_train, x_test, y_train, y_test,i):    
    accuracy = list() # accuracy of every features
    j = 0
    while(j<numberFold):
        xtr = x_train[:,i].reshape(-1,1) 
        xts = x_test[:,i].reshape(-1,1)
        classifier = RandomForestClassifier(n_estimators = 22, criterion = 'entropy', random_state = 41) # random forest classifier using information gain
        classifier.fit(xtr, y_train)
        classifierscore = classifier.score(xts, y_test) # accuracy of random forest on test data
        accuracy.append(classifierscore)
        j = j + 1
    avg = np.mean(accuracy) # take average of all accuracies
    return avg   


# In[14]:


def remainingFeatureSelect(attributes,x_train, x_test, y_train, y_test):
    score = list()
    for i in range(num_of_columns): # iterate to remaining features and add with the existing features
        if(i in attributes): # skip the feature if it already exist in list and score set to zero
            score.append(0)
            continue
        precision = list()
        attributes_copy = attributes.copy()
        attributes_copy.append(i)
        j = 0
        while(j<numberFold):
            xtr = x_train[:,attributes_copy]
            xts = x_test[:,attributes_copy]
            classifier = RandomForestClassifier()
            classifier.fit(xtr, y_train)
            classifierscore = classifier.score(xts, y_test)
            precision.append(classifierscore)
            j = j + 1
        score.append(np.mean(precision))
    arghigh = np.argmax(score) # get index of highest score of the feature
    maximum = score[arghigh] # score of highest max feature
    return maximum, arghigh


# In[15]:


def sequentialForwardSelection(partition_of_data):
    i = 0
    # set of train and test with the 5th fold of data
    while(i<numberFold):
        test_data = np.asarray(partition_of_data[i])
        x_test = test_data[:,:-1]   
        y_test = test_data[:,-1]
        array = np.arange(numberFold).tolist()
        array.remove(i)
        train_data = partition_of_data[array[0]]
        array_length = len(array)
        for a in range(1,array_length):
            tup = (train_data, partition_of_data[array[a]])
            train_data = np.concatenate(tup)
        x_train = train_data[:,:-1]
        y_train = train_data[:,-1]
        i = i + 1

    attributes = list() # list stores the features 
    score = list() # store the score of the features
    while(i<num_of_columns):
        res = firstFeatureSelect(x_train, x_test, y_train, y_test,i) # call the firstfeatureselection fuction
        score.append(res)
        i = i + 1
    attributes.append(np.argmax(score)) # take column number of highest accuracy
    
    precision_scores = list() # list stores the accuracy of all selected features
    while(len(attributes) < num_of_cols_seven_five):   # iterate to 75% features
        res2, res3 = remainingFeatureSelect(attributes,x_train, x_test, y_train, y_test) # call the reamainingfeatureselect function
        precision_scores.append(res2)
        attributes.append(res3)
    indices = np.argmax(precision_scores) # get index of highest score of the feature
    final_score = precision_scores[indices] # score of highest max feature
    best_indexes = attributes[:indices+1]
    return best_indexes,final_score


# In[16]:


final_cols, score = sequentialForwardSelection(validateData)
print("The Precision is: ", score)
print("The Best Features is: ", final_cols)


# In[ ]:




