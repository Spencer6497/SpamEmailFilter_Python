import os
import math
import numpy as np
from SPFilter_MultinomialNB import MultinomialNB_class
"""
from SPFilter_BernoulliNB import BernoulliNB_class
from SPFilter_GaussianNB import GaussianNB_class
"""
import re

test_file_path = 'test-mails'
train_file_path = 'train-mails'
# Max # of most important words
most_common_word = 3000

wordMap = {}
commonMap = []


 
#avoid 0 terms in features
smooth_alpha = 1.0

class_num =2 #we have only two classes: ham and spam
class_log_prior = [0.0, 0.0] #probability for two classes
feature_log_prob = np.zeros((class_num, most_common_word)) #feature parameterized probability
SPAM = 1 #spam class label
HAM = 0 #ham class label

#read file names in the specific file path
def read_file_names(file_path):
    return os.listdir(file_path)

#read in the specific file
def read_file(file):
    content = ''
    with open(file) as f:
        for line in f:
            line = line.strip()
            if line:
                content += line + " "
    return content.strip()

#count the total words
def count_total_word(words):
    for word in words:
        # Skip word if it doesn't contain only alphabetic characters
        if not word[0].isalpha():
            continue
        # If word doesn't appear in wordMap, add it with an occurrence of 1
        if not (word in wordMap.keys()):
            wordMap[word] = 1
        # Otherwise, increment the count of the word in the map
        else:
            count = wordMap[word]
            wordMap[word] = count+1

            
#count the word in one file
def count_word(words, singleWordMap = {}):
    for word in words:
        if not word[0].isalpha():
            continue
        if not (word in singleWordMap.keys()):
            singleWordMap[word] = 1
        else:
            count = singleWordMap[word]
            singleWordMap[word] = count+1    

#find the most common words in files store in commonMap
def most_common():
    #sort the wordMap by order of occurrence
    sort_wordMap = {k: v for k, v in sorted(wordMap.items(), key=lambda x: x[1], reverse=True)}

    #add the most common words (<= 3000 words) into commonMap
    index = 0
    for key in sort_wordMap.keys():
        if index < most_common_word:
            commonMap.append(key)
        else:
            break
        index += 1

#generate features according to commonMap, fill out feature matrix
def generate_feature(features, path, files):
    singleWordMap = {}
    file_index = 0
    for file in files:
        singleWordMap = {}  # Map already created
        content = read_file(path+'/'+file)
        #content.replace("\n", "")
        contents = content.split(" ")
        count_word(contents, singleWordMap)
        
        for key1 in singleWordMap.keys():
            common_index = 0
            for key2 in commonMap:
                if key1 == key2:
                    features[file_index][common_index] = singleWordMap[key1]
                common_index += 1
        file_index += 1


# Read in filenames of training set, store in array
files = read_file_names(train_file_path)

# Flatten email content into string, convert to list, count # words across all emails
for i in range(len(files)):
    content = read_file(train_file_path+'/'+files[i])
    #content.replace("\n", "")
    contents = content.split(" ")
    count_total_word(contents)

print("The maximum of most_common can be: ", len(wordMap))

# Pick out the 3000 most common words from the generated wordMap and add them to commonMap
most_common()

#construct model
#training feature matrix
# Rows = # files, Cols = length of commonMap (3000)
train_features = np.zeros((len(files), len(commonMap)))
# Generate occurrence of each word per file, store to training feature matrix
generate_feature(train_features, train_file_path, files)

#training labels
# Labels each email in the test set as either ham (0) or spam (1) for supervised learning
train_labels = np.zeros(len(files))
for i in range(len(files)//2, len(files)):
    train_labels[i] = 1

#verify model
#load test data
files = read_file_names(test_file_path)
#testing feature matrix
test_features = np.zeros((len(files), len(commonMap)))
generate_feature(test_features, test_file_path, files)
        
#testing labels
test_labels = np.zeros(len(files))
for i in range(len(files)//2, len(files)):
    test_labels[i] = 1


#Multinomial Naive Bayes start
#print(train_labels)
#train model
MultinomialNB = MultinomialNB_class()
# Pass training feature matrix as well as training labels to constructor
MultinomialNB.MultinomialNB(train_features, train_labels)
#test model
classes = MultinomialNB.MultinomialNB_predict(test_features)
error = 0
for i in range(len(files)):
    if test_labels[i] == classes[i]:
        error += 1
print("Multinomial Naive Bayes: ", float(error)/float(len(test_labels)))
#Multinomial Naive Bayes end

"""
#Bernoulli Naive Bayes start
BernoulliNB = BernoulliNB_class()
BernoulliNB.BernoulliNB(train_features, train_labels)
classes = BernoulliNB.BernoulliNB_predict(test_features)
error = 0
for i in range(len(files)):
    if test_labels[i] == classes[i]:
        error += 1
print("Bernoulli Naive Bayes: ", float(error)/float(len(test_labels)))
#Bernoulli Naive Bayes end

#Gaussian Naive Bayes start
GaussianNB = GaussianNB_class()
GaussianNB.GaussianNB(train_features, train_labels)
classes = GaussianNB.GaussianNB_predict(test_features)
error = 0
for i in range(len(files)):
    if test_labels[i] == classes[i]:
        error += 1
print("Gaussian Naive Bayes: ", float(error)/float(len(test_labels)))
#Gaussian Naive Bayes end
"""