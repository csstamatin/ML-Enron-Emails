#!/usr/bin/python

import sys
import pickle
import winsound
from time import time
sys.path.append("../tools/")
sys.path.append("C://Users/cssta/OneDrive/Documents/PyFiles")
from code_alarm import textDone

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import numpy as np
import matplotlib.pyplot as mp
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.feature_selection import SelectPercentile, f_classif, SelectKBest
from sklearn.pipeline import Pipeline
from tester import test_classifier
t0 = time()
### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'total_payments', "expenses",
                'total_stock_value', 'deferred_income', 'deferral_payments',
                'restricted_stock_deferred', "loan_advances",
                'exercised_stock_options', 'other', 'long_term_incentive',
                'restricted_stock', 'director_fees', 'to_messages',
                'from_poi_to_this_person', 'from_messages',
                'from_this_person_to_poi', 'shared_receipt_with_poi',
                "mes_from_poi_ratio", "mes_to_poi_ratio", "all_poi_mes_ratio"] 
                # You will need to use more features


# features_list = ['poi', 'salary', 'total_stock_value', 'deferred_income', 'exercised_stock_options', 'long_term_incentive', 'restricted_stock']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

for i in data_dict:
    totmes = data_dict[i]['from_messages'] + data_dict[i]['to_messages']
    if data_dict[i]['from_poi_to_this_person'] == "NaN" and data_dict[i]['from_this_person_to_poi'] == "NaN":
        data_dict[i]["mes_from_poi_ratio"] = "NaN"
        data_dict[i]["mes_to_poi_ratio"] = "NaN"
        data_dict[i]["all_poi_mes_ratio"] = "NaN"
    elif data_dict[i]['from_poi_to_this_person'] == "NaN":
        frrat = "NaN"
        torat = data_dict[i]['from_this_person_to_poi'] / float(data_dict[i]['to_messages'])
        allrat = torat / float(totmes)
        data_dict[i]["mes_from_poi_ratio"] = frrat
        data_dict[i]["mes_to_poi_ratio"] = torat
        data_dict[i]["all_poi_mes_ratio"] = allrat
    elif data_dict[i]['from_this_person_to_poi'] == "NaN":
        frrat = data_dict[i]['from_poi_to_this_person'] / float(data_dict[i]['from_messages'])
        torat = "NaN"
        allrat = frrat / float(totmes)
        data_dict[i]["mes_from_poi_ratio"] = frrat
        data_dict[i]["mes_to_poi_ratio"] = torat
        data_dict[i]["all_poi_mes_ratio"] = allrat
    else:
        frrat = data_dict[i]['from_poi_to_this_person'] / float(data_dict[i]['from_messages'])
        torat = data_dict[i]['from_this_person_to_poi'] / float(data_dict[i]['to_messages'])
        allrat = (frrat + torat) / float(totmes)
        data_dict[i]["mes_from_poi_ratio"] = frrat
        data_dict[i]["mes_to_poi_ratio"] = torat
        data_dict[i]["all_poi_mes_ratio"] = allrat

data_dict.pop("TOTAL", 0)
my_dataset = data_dict


### Extract features and labels from dataset for local testing
data, ppoi_names = featureFormat(my_dataset, features_list, sort_keys = True)

arr_data = []

for cnt in range(len(data[0])):
    tmp_list = []
    for peep in range(len(data)):
        tmp_list.append(data[peep][cnt])
    arr_data.append(tmp_list)


for feet in range(len(arr_data)):
    for ppoip in range(len(arr_data[feet])):
        if feet == 4:
            if ppoi_names[ppoip] == "BHATNAGAR SANJAY":
                arr_data[feet][ppoip] = 15456290.0
            elif arr_data[feet][ppoip] < 0:
                arr_data[feet][ppoip] = 0
        elif feet == 6:
            if arr_data[feet][ppoip] < 0:
                arr_data[feet][ppoip] = 0
        elif feet == 7:
            if ppoi_names[ppoip] == "BHATNAGAR SANJAY":
                arr_data[feet][ppoip] = -2604490.0
            elif arr_data[feet][ppoip] > 0:
                arr_data[feet][ppoip] = -arr_data[feet][ppoip]
        elif feet == 12:
            if arr_data[feet][ppoip] < 0:
                arr_data[feet][ppoip] = abs(arr_data[feet][ppoip])


new_data = []

for lstcnt in range(len(arr_data[0])):
    tmp2_list = []
    for ftcnt in range(len(arr_data)):
        tmp2_list.append(arr_data[ftcnt][lstcnt])
    new_data.append(tmp2_list)


scalr = MinMaxScaler()
scld_data = scalr.fit_transform(new_data)

labels, features = targetFeatureSplit(scld_data)

pcaf = PCA()
selector = SelectKBest()
kncl = KNeighborsClassifier()
dtcl = DecisionTreeClassifier()
rfcl = RandomForestClassifier()
adab = AdaBoostClassifier()
gaus = GaussianNB()
svcl = SVC()
sss = StratifiedShuffleSplit(n_splits=100, test_size=0.3, random_state=42)
estimators = [("select", pcaf), ('clf', kncl)]
pipe = Pipeline(estimators)
param_grid = dict(select__n_components=range(2, 10), 
                  clf__metric=["euclidean", "minkowski"],
                #   clf__bootstrap=[True, False])
                #   clf__n_estimators=[5, 10, 20, 30, 40, 50, 100])
                #   clf__learning_rate=[.2, .5, 1, 2, 5])
                  clf__n_neighbors=range(2,10), clf__leaf_size=[2, 5, 10, 20, 30])
grid = GridSearchCV(pipe, param_grid, scoring="f1", cv=sss)
grid.fit(features, labels)
clf = grid.best_estimator_


# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

prec, recl = test_classifier(clf, my_dataset, features_list)
# prec, recl = [.57589, .347468]
output = "Precision: %s \nRecall: %s \nTime: %s" % (str(round(prec, ndigits=4)),
         str(round(recl, ndigits=4)), round(time()-t0, 1))
# dump_classifier_and_data(clf, my_dataset, features_list)
# print output
# winsound.Beep(500, 1000)
textDone(output)