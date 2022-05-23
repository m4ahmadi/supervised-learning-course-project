# Load the dataset
import numpy as np
import urllib
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import svm, datasets, cross_validation
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import  label_binarize
from sklearn.metrics import f1_score, roc_auc_score

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


###############################################################################

#transform the data
lreg = LogisticRegression(C=1, penalty='l2', multi_class='ovr')


###############################################################################
# Set the parameters by cross-validation

validation_micro_means = list()
validation_macro_means = list()
validation_ROC_means = list()


r_param = (0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000)

for r in r_param:
    print(r)
    lreg.set_params(C=r)
    lreg.fit(trainx, trainy)
    model = SelectFromModel(lreg, prefit=True)
    trainx_new = model.transform(trainx)
    vx_new = model.transform(vx)
    print("After Feature Selection: %s_%s"
      %(trainx_new.shape, vx_new.shape))
    #Fit Model
    y_score = lreg.decision_function(vx);
    v_predict = lreg.predict(vx);
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
 
print("OVR")
print(r_param)
print(validation_micro_means)
print(validation_macro_means)
print(validation_ROC_means)


lreg = LogisticRegression(C=1, penalty='l2', multi_class='multinomial', solver='lbfgs')

for r in r_param:
    print(r)
    lreg.set_params(C=r)
    lreg.fit(trainx, trainy)
    model = SelectFromModel(lreg, prefit=True)
    trainx_new = model.transform(trainx)
    vx_new = model.transform(vx)
    print("After Feature Selection: %s_%s"
      %(trainx_new.shape, vx_new.shape))
    #Fit Model
    y_score = lreg.decision_function(vx);
    v_predict = lreg.predict(vx);
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
 
print("multinomial")
print(r_param)
print(validation_micro_means)
print(validation_macro_means)
print(validation_ROC_means)

#Best settings:

lreg = LogisticRegression(C=10, penalty='l2', multi_class='ovr')
lreg.fit(trainx, trainy)
t_predict = lreg.predict(testx);

f = h5py.File('y_test.h5','a')
dset = f.create_dataset('logistic',data=t_predict)




