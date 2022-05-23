# Load the dataset
import numpy as np
import urllib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets, cross_validation
from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier
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
trainy = dataset['mult_train']
vy = dataset['mult_val']
print("%s\n%s\n%s\n%s\n%s" %(trainx, vx, testx, trainy, vy))
print(trainy[1:10])


###############################################################################

clf = KNeighborsClassifier(algorithm='auto')

###############################################################################
# Set the parameters by cross-validation

validation_micro_means = list()
validation_macro_means = list()
validation_ROC_means = list()


#reg_param = (2, 5, 10, 50, 100, 500, 1000)
reg_param = (5, 10)

for k in reg_param:
    print(k)
    clf.set_params(n_neighbors=k)
    y_score = clf.fit(trainx, trainy).predict_proba(vx)
    # Compute cross-validation score using all CPUs
    v_predict = clf.predict(vx);
    i_score = f1_score(v_predict, vy, average='micro')
    a_score = f1_score(v_predict, vy, average='macro')
    r_score = 0
    for i in range(4999):
        r_score += sum(np.bitwise_and(np.uint64(vy[i,:]),np.uint64(v_predict[i,:])))/sum(np.bitwise_or(np.uint64(vy[i,:]), np.uint64(v_predict[i,:])))
    r_score /= vy.shape[0]
    validation_micro_means.append(i_score.mean())
    validation_micro_means.append(i_score.std())
    validation_macro_means.append(a_score.mean())
    validation_macro_means.append(a_score.std())
    validation_ROC_means.append(r_score.mean())
    validation_ROC_means.append(r_score.std())
 
print("Linear")
print(reg_param)
print(validation_micro_means)
print(validation_macro_means)
print(validation_ROC_means)



#Best settings:
clf.set_params(n_neighbors=5)
clf.fit(trainx, trainy)
t_predict = clf.predict(testx);
f = h5py.File('multi_test.h5','a')
dset = f.create_dataset('knn',data=t_predict)





