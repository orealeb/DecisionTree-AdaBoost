

import scipy.io as sio
from scipy import stats
import numpy as np
import math
import random
import decisiontree
import csv

# original author https://github.com/varunkp/SpamDecisionTree

train_filename = "./Datasets/sampled_training_Ore.csv"
test_filename = "test.csv"
training_data = decisiontree.Dataset(train_filename)
test_data = decisiontree.Dataset(test_filename, True)
target = "IsBadBuy"


array = np.array

#read in label
labels = []
for line in open("labels.csv"):
    line = line.replace('"', '').strip()
    labels.append(line)


def adaboost(training_data, rounds):
    m = training_data.leng
    weights = np.ones(m) * 1.0 / m
    strong_hypothesis = np.zeros(m)
    learners = []
    alphas = []


    for t in range(rounds):

        error = 0.0
        resampled_examples = []
        examples_index = resample(weights, m)

        for i in range(m):
            resampled_examples.append(training_data.instances[examples_index[i]])


        default = decisiontree.mode(resampled_examples, -1)
        print  "round " + str(t+1) + " training..."
        weak_learner = decisiontree.learn_decision_tree(resampled_examples, training_data.attr_names, default, target, 0) #id3.id3_depth_limited(resampled_examples, attributes, 2)
        learners.append(weak_learner)

        classifications = decisiontree.classify(training_data.instances, weak_learner)
        error = 0
        for i in range(len(classifications)):
            predicted = classifications[i]
            error += (predicted != resampled_examples[i][-1])*weights[i]
        
        train_accuracy = decisiontree.tree_accuracy(resampled_examples, weak_learner)
        print "Training Accuracy with sampled dataset= " + str(train_accuracy) + "%"
         
        train_accuracy = decisiontree.tree_accuracy(training_data.instances, weak_learner)
        print "Training Accuracy with training dataset= " + str(train_accuracy) + "%"

        print "Error", error

        if error == 0.0:
            alpha = 4.0
        elif error > 0.5:
            print    "Discarding learner"
            continue    #discard classifier with error > 0.5
        else:
            alpha = 0.5 * np.log((1 - error)/error)

        alphas.append(alpha)
        learners.append(weak_learner)
        print  "weak learner added"

        for i in range(m):
            y = resampled_examples[i][-1]
            h = classifications[i]
            h = -1 if h == 0 else 1
            y = -1 if y == 0 else 1
            weights[i] = weights[i] * np.exp(-alpha * h * y)
        sum_weights = sum(weights)
        print 'Sum of weights', sum_weights
        normalized_weights = [float(w)/sum_weights for w in weights]
        weights = normalized_weights
    
    return zip(alphas, learners)


def classify(strong_hypothesis, example):
    classification = 0
    for weight, learner in strong_hypothesis:
        if str(learner.predict(example)) == str(1):
            ex_class = 1  
        else:
            ex_class = -1
        classification += weight*ex_class
        #print str(learner.predict(example)) +" "+ str(classification) + " "+str(weight) + " "+ str(ex_class)

    return 1 if classification > 0 else 0



def resample(weights, m):
    xk = np.arange(m)
    pk = weights
    custm = stats.rv_discrete(name='custm', values=(xk, pk))
    R = custm.rvs(size=m)
    return R



ret = adaboost(training_data, 5)
print "adaboost building complete!"

print ret

predictions = []

for row in test_data.instances:
    predictions.append(classify(ret, row))

i = 1
for weight, learner in ret:
    train_accuracy = decisiontree.tree_accuracy(training_data.instances, learner)
    print "Training Accuracy for learner " + str(i) + ", weight " + str(weight) + " is " + str(train_accuracy) + "%"
    i+=1

res_file = open('adaboost_res.csv', 'w')
writer = csv.writer(res_file)
header = 'RefId','IsBadBuy'
writer.writerow(header)  
for i in range(len(predictions)):
    writer.writerow([labels[i], predictions[i]])
res_file.close()


