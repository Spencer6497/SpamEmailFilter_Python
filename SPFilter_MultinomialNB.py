import os
import math
import numpy as np


most_common_word = 3000
#avoid 0 terms in features
smooth_alpha = 1.0
class_num =2 #we have only two classes: ham and spam
class_log_prior = [0.0, 0.0]#probability for two classes
feature_log_prob = np.zeros((class_num, most_common_word))#feature parameterized probability
SPAM = 1 #spam class label
HAM = 0 #ham class label


class MultinomialNB_class:

    #multinomial naive bayes
    # Class constructor
    def MultinomialNB(self, features, labels):
        ham = 0
        spam = 0
        for label in labels:
            if label == 0.0:
                ham += 1
            elif label == 1.0:
                spam += 1

        # calculate class_log_prior
        class_log_prior[0] = math.log(ham/len(labels))
        class_log_prior[1] = math.log(spam/len(labels))

        # Calculate feature_log_prob
        # Create ham and spam arrays
        hamArray = np.zeros(most_common_word)
        spamArray = np.zeros(most_common_word)

        # Create sum of ham and spam
        sumHam = 0
        sumSpam = 0

        for row in range(len(features)):
            for col in range(most_common_word):
                if row < (len(labels)//2):
                    hamArray[col] += features[row][col]
                    sumHam += 1
                else:
                    spamArray[col] += features[row][col]
                    sumSpam += 1

        for i in range(most_common_word):
            hamArray[i] += smooth_alpha
            spamArray[i] += smooth_alpha

        sumHam += most_common_word * smooth_alpha
        sumSpam += most_common_word * smooth_alpha

        for j in range(most_common_word):
            feature_log_prob[0][j] = math.log(hamArray[j] / sumHam)
            feature_log_prob[1][j] = math.log(spamArray[j] / sumSpam)

    #multinomial naive bayes prediction
    def MultinomialNB_predict(self, features):
        classes = np.zeros(len(features))

        ham_prob = 0.0
        spam_prob = 0.0

        # Iterate through files
        for i in range(len(features)):
            # Store temporary sums for iterative purposes
            hamTempSum = 0
            spamTempSum = 0
            # Iterate through words
            for j in range(len(features[i])):
                # Compute elementwise product
                hamTempSum += feature_log_prob[0][j] * features[i][j]
                spamTempSum += feature_log_prob[1][j] * features[i][j]

            # Calculate ham_prob and spam_prob
            ham_prob = hamTempSum + class_log_prior[0]
            spam_prob = spamTempSum + class_log_prior[1]

            # If it is more than likely ham, classify it as such and vice versa
            if ham_prob > spam_prob:
                classes[i] = 0
            else:
                classes[i] = 1

        return classes
