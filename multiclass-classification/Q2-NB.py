# Load the dataset
import numpy as np
import urllib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets, cross_validation
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import  label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif

#Loading the main datasets...
import h5py
dataset = h5py.File('../ml_proj1_data.h5', 'r')
print(dataset.keys())


#Extract the general x values
trainx = dataset['xtrain']
vx = dataset['xval']
testx = dataset['xtest']

#Extract the data for this part
trainy = dataset['ytrain']
vy = dataset['yval']
print("%s\n%s\n%s\n%s\n%s" %(trainx, vx, testx, trainy, vy))
#Be sure about the labels
myset = set(trainy)
print(myset)



###############################################################################

clf = GaussianNB()


#transform the data
#lsvc = ExtraTreesClassifier().fit(trainx, trainy)
#model = SelectFromModel(lsvc, prefit=True)
lsvc =  SelectPercentile(f_classif, percentile=50).fit(trainx, trainy).get_support(indices=True)
trainx_new = trainx[:,lsvc]
vx_new = vx[:,lsvc]
testx_new = testx[:,lsvc]
print("After Feature Selection: %s_%s_%s"
      %(trainx_new.shape, vx_new.shape, testx_new.shape))


#clf_fs = Pipeline([  ('fs', transform),('clf', clf)])


###############################################################################
# Set the parameters by cross-validation

validation_micro_means = list()
validation_macro_means = list()
validation_ROC_means = list()

y_score = clf.fit(trainx_new, trainy).predict_proba(vx_new)
# Compute cross-validation score using all CPUs
v_predict = clf.predict(vx_new);
i_score = f1_score(v_predict, vy, average='micro')
a_score = f1_score(v_predict, vy, average='macro')
r_score = roc_auc_score(label_binarize(vy, classes=range(43)),
                        y_score, average='micro')
validation_micro_means.append(i_score.mean())
validation_micro_means.append(i_score.std())
validation_macro_means.append(a_score.mean())
validation_macro_means.append(a_score.std())
validation_ROC_means.append(r_score.mean())
validation_ROC_means.append(r_score.std())
 
print("NB")
print(validation_micro_means)
print(validation_macro_means)
print(validation_ROC_means)



#Best settings:
clf = GaussianNB()
clf.fit(trainx_new, trainy)
t_predict = clf.predict(testx_new);
f = h5py.File('y_test.h5','a')
#dset = f.create_dataset('nb',data=t_predict)






