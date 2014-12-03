

import scipy.io as sio
from scipy import stats
import numpy as np
import math
import random
import decisiontree
import csv

# original author https://github.com/varunkp/SpamDecisionTree

train_filename = "./Datasets/66_SAMPLED.csv"
test_filename = "./Datasets/66_test.csv"
training_data = decisiontree.Dataset(train_filename)
test_data = decisiontree.Dataset(test_filename, True)
target = "IsBadBuy"


array = np.array

#read in label
labels = []
for line in open("labels.csv"):
    line = line.replace('"', '').strip()
    labels.append(line)


def bagging(training_data, rounds):
    m = training_data.leng
    weights = np.ones(m) * 1.0 / m
    strong_hypothesis = np.zeros(m)
    learners = []
    alphas = []


    for t in range(rounds):

        resampled_examples = []
        examples_index = random.sample(list(range(0,m)), m)

        for i in range(m):
            resampled_examples.append(training_data.instances[examples_index[i]])


        default = decisiontree.mode(resampled_examples, -1)
        print  "round " + str(t+1) + " training..."
        weak_learner = decisiontree.learn_decision_tree(resampled_examples, training_data.attr_names, default, target, 0) #id3.id3_depth_limited(resampled_examples, attributes, 2)
        learners.append(weak_learner)

    return learners


def classify(strong_hypothesis, example):
    classification = 0
    vote1 = 0
    vote0 = 0
    for learner in strong_hypothesis:
        if str(learner.predict(example)) == str(1):
            vote1 += 1  
        else:
            vote0 += 1
    return 1 if vote1 > vote0 else 0



def resample(weights, m):
    xk = np.arange(m)
    pk = weights
    custm = stats.rv_discrete(name='custm', values=(xk, pk))
    R = custm.rvs(size=m)
    return R



ret = bagging(training_data, 5)
print "bagging procedure building complete!"

print ret

predictions = []

for row in test_data.instances:
    predictions.append(classify(ret, row))



res_file = open('bagging_res.csv', 'w')
writer = csv.writer(res_file)
header = 'RefId','IsBadBuy'
writer.writerow(header)  
for i in range(len(predictions)):
    writer.writerow([labels[i], predictions[i]])
res_file.close()


