DecisionTree-AdaBoost
=====================

To use sampled_training_getkicked.csv:

--1 change line 13 and 14 in adaboost.py and line 793 and 795 in decisiontree.py
to 

train_filename = "./Datasets/sampled_training_getkicked.csv"
test_filename = "test_getkicked.csv"

--2 In addition, since sampled_getkicked.csv dataset has more attributes than sampled_Ore.csv you will have to change the numeric attributes in decisiontree.py on line 108,323, and 528 from 


[False,False,False,True,False,False,False,False,False,False,False,False,True,False,False,False,True,True,True,True,True,True,True,True,False,False,False,False,False,True,False,True,False,False]


to:



[True,True,True
	,True,True,True,True
	,True,True,True,True
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False,False,False
	,False,False
	]
