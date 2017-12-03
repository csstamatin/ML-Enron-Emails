#!/usr/bin/python

import sys
import pickle
import winsound
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

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
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


toscl_data = arr_data[1:]
poi_data = arr_data[:1]
scalr = MinMaxScaler()
fstsc_data = scalr.fit_transform(arr_data)
for scset in fstsc_data:
    poi_data.append(scset)
scld_data = poi_data

new_data = []

for lstcnt in range(len(arr_data[0])):
    tmp2_list = []
    for ftcnt in range(len(arr_data)):
        tmp2_list.append(arr_data[ftcnt][lstcnt])
    new_data.append(tmp2_list)

# print len(features_list)
# print len(data[0])
# print len(new_data[0])
# print features_list[18]
# print "min", min(arr_data[18])
# print "max", max(arr_data[18])
# mp.boxplot(arr_data[18])
# mp.show()

# print data[0]
# print new_data[0]

# for bplot in arr_data[1:]:
#     mp.boxplot(bplot)
#     mp.show()

# scalr = MinMaxScaler()
# scld_data = scalr.fit_transform(new_data)

labels, features = targetFeatureSplit(new_data)

pcaf = PCA()
selector = SelectKBest()
kncl = KNeighborsClassifier()
dtcl = DecisionTreeClassifier()
rfcl = RandomForestClassifier()
adab = AdaBoostClassifier()
gaus = GaussianNB()
svcl = SVC()
sss = StratifiedShuffleSplit(n_splits=100, test_size=0.3, random_state=42)
estimators = [("select", pcaf), ('clf', adab)]
pipe = Pipeline(estimators)
param_grid = dict(select__n_components=range(3, 10), clf__n_estimators=[30, 40, 50, 60, 70],
                  clf__learning_rate=[1, 2, 5], clf__algorithm=["SAMME", "SAMME.R"])
                #   clf__n_neighbors=range(5,10), clf__leaf_size=[30, 40 ,50])
grid = GridSearchCV(pipe, param_grid, scoring="recall", cv=sss)
grid.fit(features, labels)
clf = grid.best_estimator_
print grid.best_estimator_


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.


# pca = PCA(n_components=5)
# pca = pca.fit(features_train)
# new_train = pca.transform(features_train)
# new_test = pca.transform(features_test)

# new_train = features_train
# new_test = features_test



# features_train = features_train[:150]
# labels_train   = labels_train[:150]




# dtcl = DecisionTreeClassifier()
# dtcl.fit(features_train, labels_train)
# # print "Number of training features"
# # print len(features_train)
# # print "Length of features"
# # print len(features_list)
# print dtcl.feature_importances_
# print len(dtcl.feature_importances_)
# print len(features_list)
# new_feats2 = []
# for y in range(len(dtcl.feature_importances_)):
#     if dtcl.feature_importances_[y] > .1:
#         new_feats2.append(features_list[y+1])
# print new_feats2

# pred = dtcl.predict(features_test)
# acc = accuracy_score(pred, labels_test)
# print acc

# selector2 = SelectKBest(f_classif, k=5)
# selector2.fit(features, labels)
# new_feats3 = []
# for feet3 in selector2.get_support(indices=True):
#     new_feats3.append(features_list[feet3])
# # print new_feats3

# selector = SelectPercentile(f_classif, percentile=30)
# selector.fit(features, labels)
# new_feats = []
# for feet in selector.get_support(indices=True):
#     new_feats.append(features_list[feet+1])
# print new_feats

# ftft_list = [new_feats, new_feats2, new_feats3]
# nwfeat_list = ["poi",]
# for p in ftft_list:
#     for t in p:
#         if t in nwfeat_list or t == "poi":
#             pass
#         else:
#             nwfeat_list.append(t)

# print len(nwfeat_list)
# print nwfeat_list


# imp_list = []
# clf10 = AdaBoostClassifier(n_estimators=50)
# clf10 = clf10.fit(features_train, labels_train)
# pred10 = clf10.predict(features_test)
# acc10 = accuracy_score(pred10, labels_test)
# print acc10

# parameters = {'n_estimators':[2, 4, 7, 10, 15, 20, 30, 40], "criterion":("gini", "entropy"),
#               "min_samples_split":[2, 4, 7, 10, 15, 20, 30, 40]}
# rfc = RandomForestClassifier()
# clfg = GridSearchCV(rfc, parameters).fit(new_train, labels_train)
# print clfg.best_params_

# clf2 = RandomForestClassifier(min_samples_split=20, n_estimators=4)
# clf2 = clf2.fit(features_train, labels_train)
# pred2 = clf2.predict(features_test)
# acc2 = accuracy_score(pred2, labels_test)
# psion2 = precision_score(labels_test, pred2)
# rcal2 = recall_score(labels_test, pred2)
# print acc2, rcal2, psion2

# clf7 = RandomForestClassifier(min_samples_split=10, n_estimators=10)
# clf7 = clf7.fit(new_train, labels_train)
# pred7 = clf7.predict(new_test)
# acc7 = accuracy_score(pred7, labels_test)
# psion7 = precision_score(labels_test, pred7)
# rcal7 = recall_score(labels_test, pred7)
# print acc7, rcal7, psion7

# clf3 = KNeighborsClassifier(n_neighbors=25)
# clf3 = clf3.fit(features_train, labels_train)
# pred3 = clf3.predict(features_test)
# acc3 = accuracy_score(pred3, labels_test)
# print acc3

# svc1 = tree.r
# clf5 = GridSearchCV(svc1, parameters)
# clf5 = clf5.fit(features_train, labels_train)
# print clf5.best_params_

# clf4 = DecisionTreeClassifier(min_samples_split=40)
# clf4 =clf4.fit(features_train, labels_train)
# pred4 = clf4.predict(features_test)
# acc4 = accuracy_score(pred4, labels_test)
# print acc4

# 
# parameters = {'kernel':('rbf',), 'C':[.0001, .001, .01, .1, 1, 10, 50, 100, 1000, 10000, 100000, 1000000],
#              "gamma":[.00001, .0001, .001, .01, .1, 1, 10, 100, 1000, 10000]}
# svc1 = SVC()
# clf5 = GridSearchCV(svc1, parameters)
# clf5 = clf5.fit(features_train, labels_train)
# print clf5.best_params_

# pred5 = clf5.predict(features_test)
# acc5 = accuracy_score(pred5, labels_test)
# prec5 = precision_score(labels_test, pred5)
# rcal5 = recall_score(labels_test, pred5)
# print "Accuracy"
# print acc5, rcal5, prec5

# clf = SVC(kernel='rbf', C=50, gamma=1)
# clf = clf.fit(features_train, labels_train)
# pred5 = clf.predict(features_test)
# acc5 = accuracy_score(pred5, labels_test)
# # prec5 = precision_score(labels_test, pred5)
# rcal5 = recall_score(labels_test, pred5)
# print "Accuracy"
# print acc5, rcal5

# clf6 = GaussianNB()
# clf6 = clf6.fit(features_train, labels_train)
# pred6 = clf6.predict(features_test)
# acc6 = accuracy_score(pred6, labels_test)
# print acc6

x = test_classifier(clf, my_dataset, features_list)
print x
# dump_classifier_and_data(clf, my_dataset, features_list)

winsound.Beep(500, 1000)
textDone(x)