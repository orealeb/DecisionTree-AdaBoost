DecisionTree-AdaBoost
=====================

The program currently works with './Datasets/66_SAMPLED.csv' and './Datasets/66_test.csv'.

To run: 
All the files are already in there expected directories, therefore, to run simply enter the algorithm's name i.e *python algorithmname.py*. 

1. preprocessing.py
  * input file: datasets folder containing training.csv, test.csv
  * output file: training_new.csv, test_new.csv'
  
For the classifier algorithms the output file is already formatted with IsBadBuy predictions and refIDs, so any of decision_res.csv, bagging_res.csv, and adaboost_res.csv can be submitted to Kaggle to obtain a score.
  
2. decisiontree.py
  * input file: datasets folder containing 66_SAMPLED.csv, 66_test.csv, and ids.csv
  * output file: decision_tree_res.csv

3. bagging.py
  * input file: datasets folder containing 66_SAMPLED.csv, 66_test.csv, and ids.csv
  * output file: bagging_res.csv

4. adaboost.py
  * input file: datasets folder containing 66_SAMPLED.csv, 66_test.csv, and ids.csv
  * output file: adaboost_res.csv
  
Currently bagging.py and adaboost.py is set to a default 5 number of rounds. To change this, visit line 183 for adaboost.py or 134 for bagging.py, containing the line:
ret = train(training_data, attr_names, label_name, num_attr, 5)
