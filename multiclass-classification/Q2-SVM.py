# Load the dataset
import numpy as np
import urllib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets, cross_validation
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import  label_binarize
from sklearn.multiclass import OneVsRestClassifier

#Loading the main datasets...
import h5py
dataset = h5py.File('../ml_proj1_data.h5', 'r')
print(dataset.keys())


#Extract the general x values
trainx = dataset['xtrain']
vx = dataset['xval']
testx = dataset['xtest']

#Extract the data for this part
trainy_old = dataset['ytrain']
vy = dataset['yval']
print("%s\n%s\n%s\n%s\n%s" %(trainx, vx, testx, trainy_old, vy))
#Be sure about the labels
myset = set(trainy_old)
print(myset)

#make y values binary in a ovo scheme
trainy = label_binarize(trainy_old, classes=range(43))
vy = label_binarize(vy, classes=range(42))

###############################################################################



#transform the data
lsvc = LinearSVC(C=1, penalty="l1", dual=False).fit(trainx, trainy_old)
model = SelectFromModel(lsvc, prefit=True)
trainx_new = model.transform(trainx)
vx_new = model.transform(vx)
testx_new = model.transform(testx)
print("After Feature Selection: %s_%s_%s"
      %(trainx_new.shape, vx_new.shape, testx_new.shape))


#clf_fs = Pipeline([  ('fs', transform),('clf', clf)])


###############################################################################
# Set the parameters by cross-validation

validation_micro_means = list()
validation_macro_means = list()
validation_ROC_means = list()


reg_param = (0.001, 0.01, 1, 10, 100)

for reg in reg_param:
    print(reg)
    clf = OneVsRestClassifier(svm.SVC(kernel='rbf', C=reg, probability=True))
    y_score = clf.fit(trainx_new, trainy).decision_function(vx_new)
    # Compute cross-validation score using all CPUs
    v_predict = clf.predict(vx_new);
    i_scores = f1_score(v_predict, vy, average='micro')
    a_scores = accuracy_score(v_predict, vy, average='macro')
    r_score = roc_auc_score(v_predict, y_score, average='micro')
    print (f_scores.mean())    
    validation_micro_means.append(i_scores.mean())
    validation_micro_means.append(i_scores.std())
    validation_macro_means.append(a_scores.mean())
    validation_macro_means.append(a_scores.std())
    validation_ROC_means.append(r_scores.mean())
    validation_ROC_means.append(r_scores.std())
 
print("Linear")
print(reg_param)
print(validation_micro_means)
print(validation_macro_means)
print(validation_ROC_means)



#Best settings:
clf.set_params(kernel='rbf', gamma='auto', C=100)
clf.fit(trainx_new, trainy)
t_predict = clf.predict(testx_new);

f = h5py.File('y_test.h5','w')
dset = f.create_dataset('svm',(3000,),data=t_predict)





