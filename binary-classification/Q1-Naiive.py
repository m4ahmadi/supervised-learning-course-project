# Load the dataset
import numpy as np
import urllib
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import svm, datasets, cross_validation
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score, accuracy_score

#Loading the main datasets...
import h5py
dataset = h5py.File('../ml_proj1_data.h5', 'r')
print(dataset.keys())


#Extract the general x values
trainx = dataset['xtrain']
vx = dataset['xval']
testx = dataset['xtest']

#Extract the data for this part
trainy = dataset['bin_train']
vy = dataset['bin_val']
print("%s\n%s\n%s\n%s\n%s" %(trainx, vx, testx, trainy, vy))


###############################################################################

#transform = SelectFromModel(LinearSVC(penalty="l1", dual=False, ))
clf = GaussianNB()

#transform the data
lsvc = LinearSVC(C=1, penalty="l1", dual=False)


#clf_fs = Pipeline([  ('fs', transform),('clf', clf)])


###############################################################################
# Set the parameters by cross-validation

validation_means = list()
validation_stds = list()


validation_accmeans = list()
validation_accstds = list()

r_param = (0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000)

for r in r_param:
    print(r)
    lsvc.set_params(C=r)
    lsvc.fit(trainx, trainy)
    model = SelectFromModel(lsvc, prefit=True)
    trainx_new = model.transform(trainx)
    vx_new = model.transform(vx)
    print("After Feature Selection: %s_%s"
      %(trainx_new.shape, vx_new.shape))
    #Fit Model
    clf.fit(trainx_new, trainy)
    v_predict = clf.predict(vx_new);
    f_scores = f1_score(v_predict, vy)
    a_scores = accuracy_score(v_predict, vy)
    print (f_scores.mean())    
    validation_means.append(f_scores.mean())
    validation_means.append(f_scores.std())
    validation_accmeans.append(a_scores.mean())
    validation_accmeans.append(a_scores.std())
 
print("Gaussian")
print(r_param)
print(validation_means)
print(validation_accmeans)


#Best settings:
lsvc.set_params(C=1)
lsvc.fit(trainx, trainy)
model = SelectFromModel(lsvc, prefit=True)
trainx_new = model.transform(trainx)
testx_new = model.transform(testx)

clf = GaussianNB()
clf.fit(trainx_new, trainy)
t_predict = clf.predict(testx_new);

f = h5py.File('bin_test.h5','a')
dset = f.create_dataset('naiive',data=t_predict)




