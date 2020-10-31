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
        # calculate class_log_prior
        ham = 0
        spam = 0
        for label in labels:
            if label == 0.0:
                ham += 1
            elif label == 1.0:
                spam += 1

        class_log_prior[0] = float(math.log(ham))
        class_log_prior[1] = float(math.log(spam))

	'''
		//calculate feature_log_prob
		/**
		 * nested loop over features
		 * for row = features.length
		 *     for col = most_common
		 *         ham[col] + features[row][col]
		 *         spam[col] + features[row][col]
		 *         sum of ham
		 *         sum of spam
		 * for i = most_common
		 *     ham[i] + smooth_alpha
		 *     spam[i] + smooth_alpha
		 * sum of ham += most_common*smooth_alpha
		 * sum of spam += most_common*smooth_alpha
		 * for j = most_common
		 *     feature_log_prob[0] = Math.log(ham[i]/sum of ham)
		 *     feature_log_prob[1] = Math.log(spam[i]/sum of spam)
		 */
    '''

    #multinomial naive bayes prediction
    def MultinomialNB_predict(self, features):
        classes = np.zeros(len(features))

        ham_prob = 0.0
        spam_prob = 0.0
        '''/**
		 * nested loop over features with i and j
		 * calculate ham_prob and spam_prob
		 * add ham_prob and spam_prob with class_log_prior
		 * if ham_prob > spam_prob
		 * HAM
		 * else SPAM
		 * return  classes
		 */'''

        return classes
