#!/usr/bin/python
# -*- coding: utf-8 -*-
import scipy.io as sio
from scipy import stats
import numpy as np
import math
import random
import decisiontree
import csv

train_filename = './Datasets/66_SAMPLED.csv'
test_filename = './Datasets/66_test.csv'
training_data = decisiontree.Dataset(train_filename)
test_data = decisiontree.Dataset(test_filename, True)
label_name = 'IsBadBuy'

# read in label

labels = []
for line in open('labels.csv'):
    line = line.replace('"', '').strip()
    labels.append(line)


def bagging(training_data, rounds):
    num_rows = training_data.leng
    weights = np.ones(num_rows) * 1.0 / num_rows
    classifiers = []
    alphas = []
    for t in range(rounds):
        resampled_entries = []
        random_indices = random.sample(list(range(0, num_rows)),
                num_rows)
        for i in range(num_rows):
            resampled_entries.append(training_data.instances[random_indices[i]])
        default = decisiontree.mode(resampled_entries, -1)
        print 'round ' + str(t + 1) + ' training...'
        weak_classifier = \
            decisiontree.learn_decision_tree(resampled_entries,
                training_data.attr_names, default, label_name, 0)
        classifiers.append(weak_classifier)
    return classifiers


def classify(classifiers, example):
    classification = 0
    vote1 = 0
    vote0 = 0
    for weak_classifier in classifiers:
        if str(weak_classifier.predict(example)) == str(1):
            vote1 += 1
        else:
            vote0 += 1
    return (1 if vote1 > vote0 else 0)


classifers = bagging(training_data, 5)
print 'bagging procedure building complete!'

predictions = []

for row in test_data.instances:
    predictions.append(classify(classifers, row))

res_file = open('bagging_res.csv', 'w')
writer = csv.writer(res_file)
header = ('RefId', 'IsBadBuy')
writer.writerow(header)
for i in range(len(predictions)):
    writer.writerow([labels[i], predictions[i]])
res_file.close()
