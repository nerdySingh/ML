#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)



### your code below
values =[]
for point in data:
    salary = point[0]
    bonus = point[1]
    values.append((salary,bonus))
    #matplotlib.pyplot.scatter(salary,bonus)
outlier = sorted(values,key=lambda x:x[0],reverse=True)[0]
#print(outlier[0])

keys = 0
for key in data_dict:
    if outlier[0] == data_dict[key]['salary']:
        #print(data_dict[key])
        keys=key

data_dict.pop(keys,0)
values1 =[]
data = featureFormat(data_dict, features)
for point in data:
    salary = point[0]
    bonus = point[1]
    values1.append((salary,bonus))
    matplotlib.pyplot.scatter(salary,bonus)



outliers1 = sorted(values1,key=lambda x:x[0],reverse=True)
print(outliers1)

for key in data_dict:
    for x in range(2):
        if outliers1[x][0] == data_dict[key]['salary']:
            print(key)
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
#matplotlib.pyplot.show()


